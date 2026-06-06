# LLM 클라이언트 설정 선택 로직을 검증하는 테스트
import os
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class LlmTest(unittest.TestCase):

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

    def test_openai_workflow_model_routing_uses_fast_next_and_quality_feedback_models(self):
        from app.config import Settings
        from app.core.llm import fallback_model_for_workflow, model_for_workflow

        settings = Settings(openai_api_key="test-key")

        self.assertEqual(model_for_workflow("next_question", settings), "gpt-4o-mini")
        self.assertEqual(model_for_workflow("turn_feedback", settings), "gpt-5.4-mini")
        self.assertEqual(model_for_workflow("session_feedback", settings), "gpt-5.4-mini")
        self.assertIsNone(fallback_model_for_workflow("next_question", settings))
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
