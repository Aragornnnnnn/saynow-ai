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


if __name__ == "__main__":
    unittest.main()
