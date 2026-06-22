# LLM 클라이언트 설정 선택 로직을 검증하는 테스트
import os
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class LlmTest(unittest.TestCase):

    def test_chat_uses_max_completion_tokens_for_gpt5_models(self):
        from app.core import llm

        captured_kwargs = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)

                class Message:
                    content = '{"ok":true}'

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        class FakeClient:
            class Chat:
                completions = FakeCompletions()

            chat = Chat()

        original_client = llm.client
        try:
            llm.client = FakeClient()

            result = llm.chat(
                "Return JSON only.",
                'Return {"ok":true}.',
                max_tokens=64,
                temperature=0,
                model="gpt-5.4-mini",
            )
        finally:
            llm.client = original_client

        self.assertEqual(result, '{"ok":true}')
        self.assertEqual(captured_kwargs["model"], "gpt-5.4-mini")
        self.assertEqual(captured_kwargs["max_completion_tokens"], 64)
        self.assertNotIn("max_tokens", captured_kwargs)

    def test_chat_keeps_max_tokens_for_gpt4o_models(self):
        from app.core import llm

        captured_kwargs = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)

                class Message:
                    content = '{"ok":true}'

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        class FakeClient:
            class Chat:
                completions = FakeCompletions()

            chat = Chat()

        original_client = llm.client
        try:
            llm.client = FakeClient()

            llm.chat(
                "Return JSON only.",
                'Return {"ok":true}.',
                max_tokens=64,
                temperature=0,
                model="gpt-4o-mini",
            )
        finally:
            llm.client = original_client

        self.assertEqual(captured_kwargs["model"], "gpt-4o-mini")
        self.assertEqual(captured_kwargs["max_tokens"], 64)
        self.assertNotIn("max_completion_tokens", captured_kwargs)

    def test_openai_client_uses_configured_request_timeout(self):
        from app.core import llm

        self.assertEqual(llm.client.timeout, llm.settings.llm_request_timeout_seconds)

    def test_resolve_llm_options_uses_upstage_provider(self):
        from app.config import Settings
        from app.core.llm import resolve_llm_options

        settings = Settings(
            llm_provider="upstage",
            upstage_api_key="test-upstage-key",
            upstage_model="solar-pro3",
        )

        options = resolve_llm_options(settings)

        self.assertEqual(options.api_key, "test-upstage-key")
        self.assertEqual(options.base_url, "https://api.upstage.ai/v1")
        self.assertEqual(options.model, "solar-pro3")

    def test_resolve_llm_options_uses_openrouter_provider(self):
        from app.config import Settings
        from app.core.llm import fallback_model_for_workflow, model_for_workflow, resolve_llm_options

        settings = Settings(
            llm_provider="openrouter",
            openrouter_api_key="test-openrouter-key",
        )

        options = resolve_llm_options(settings)

        self.assertEqual(options.api_key, "test-openrouter-key")
        self.assertEqual(options.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(options.model, "openai/gpt-5.4-mini")
        self.assertEqual(model_for_workflow("turn_feedback", settings), "openai/gpt-5.4-mini")
        self.assertIsNone(fallback_model_for_workflow("turn_feedback", settings))

    def test_openai_workflow_model_routing_uses_quality_models_for_all_user_facing_workflows(self):
        from app.config import Settings
        from app.core.llm import fallback_model_for_workflow, model_for_workflow

        settings = Settings(openai_api_key="test-key")

        self.assertEqual(model_for_workflow("next_question", settings), "gpt-5.4-mini")
        self.assertEqual(model_for_workflow("turn_feedback", settings), "gpt-5.4-mini")
        self.assertEqual(model_for_workflow("session_feedback", settings), "gpt-5.4-mini")
        self.assertEqual(fallback_model_for_workflow("next_question", settings), "gpt-4o-mini")
        self.assertEqual(fallback_model_for_workflow("turn_feedback", settings), "gpt-4o-mini")
        self.assertEqual(fallback_model_for_workflow("session_feedback", settings), "gpt-4o-mini")

    def test_non_openai_workflow_model_routing_uses_provider_model_without_openai_fallback(self):
        from app.config import Settings
        from app.core.llm import fallback_model_for_workflow, model_for_workflow

        settings = Settings(
            llm_provider="upstage",
            upstage_api_key="test-upstage-key",
            upstage_model="solar-pro3",
        )

        self.assertEqual(model_for_workflow("turn_feedback", settings), "solar-pro3")
        self.assertIsNone(fallback_model_for_workflow("turn_feedback", settings))


if __name__ == "__main__":
    unittest.main()
