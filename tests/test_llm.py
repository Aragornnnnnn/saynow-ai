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


if __name__ == "__main__":
    unittest.main()
