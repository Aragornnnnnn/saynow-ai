# 설정 객체가 배포 환경변수를 안전하게 읽는지 검증하는 테스트
import os
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class ConfigTest(unittest.TestCase):

    def test_settings_ignore_unrelated_deploy_environment_values(self):
        from app.config import Settings

        settings = Settings(
            openai_api_key="test-key",
            db_url="postgresql://example",
            saynow_ai_base_url="http://example.com",
        )

        self.assertEqual(settings.openai_api_key, "test-key")

    def test_settings_support_upstage_provider_values(self):
        from app.config import Settings

        settings = Settings(
            llm_provider="upstage",
            upstage_api_key="test-upstage-key",
            upstage_model="solar-pro3",
        )

        self.assertEqual(settings.llm_provider, "upstage")
        self.assertEqual(settings.upstage_api_key, "test-upstage-key")
        self.assertEqual(settings.upstage_base_url, "https://api.upstage.ai/v1")
        self.assertEqual(settings.upstage_model, "solar-pro3")

    def test_settings_support_sentry_observability_values(self):
        from app.config import Settings

        settings = Settings(
            sentry_dsn="https://public@example.ingest.sentry.io/123",
            sentry_environment="develop",
            sentry_traces_sample_rate=0.25,
            sentry_max_breadcrumbs=150,
        )

        self.assertEqual(settings.sentry_dsn, "https://public@example.ingest.sentry.io/123")
        self.assertEqual(settings.sentry_environment, "develop")
        self.assertEqual(settings.sentry_traces_sample_rate, 0.25)
        self.assertEqual(settings.sentry_max_breadcrumbs, 150)

    def test_assistance_rag_is_disabled_by_default(self):
        from app.config import Settings

        settings = Settings(openai_api_key="test-key")

        self.assertFalse(settings.assistance_rag_enabled)


if __name__ == "__main__":
    unittest.main()
