# Sentry 초기화와 예외 캡처 헬퍼를 검증하는 테스트
from types import SimpleNamespace
import logging
import unittest


class FakeSentrySdk:
    def __init__(self):
        self.init_kwargs = None
        self.captured_exceptions = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def capture_exception(self, exc):
        self.captured_exceptions.append(exc)


class FakeFastApiIntegration:
    def __init__(self):
        pass


class FakeLoggingIntegration:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ObservabilityTest(unittest.TestCase):

    def test_init_sentry_skips_when_dsn_is_missing(self):
        from app.core.observability import init_sentry

        fake_sentry = FakeSentrySdk()
        config = SimpleNamespace(
            sentry_dsn=None,
            sentry_environment="local",
            sentry_traces_sample_rate=0.0,
            sentry_max_breadcrumbs=100,
        )

        initialized = init_sentry(config, sentry_sdk_module=fake_sentry)

        self.assertFalse(initialized)
        self.assertIsNone(fake_sentry.init_kwargs)

    def test_init_sentry_uses_configured_dsn_and_environment(self):
        from app.core.observability import init_sentry

        fake_sentry = FakeSentrySdk()
        config = SimpleNamespace(
            sentry_dsn="https://public@example.ingest.sentry.io/123",
            sentry_environment="develop",
            sentry_traces_sample_rate=0.25,
            sentry_max_breadcrumbs=150,
        )

        initialized = init_sentry(
            config,
            sentry_sdk_module=fake_sentry,
            fastapi_integration_cls=FakeFastApiIntegration,
            logging_integration_cls=FakeLoggingIntegration,
        )

        self.assertTrue(initialized)
        self.assertEqual(fake_sentry.init_kwargs["dsn"], config.sentry_dsn)
        self.assertEqual(fake_sentry.init_kwargs["environment"], "develop")
        self.assertEqual(fake_sentry.init_kwargs["traces_sample_rate"], 0.25)
        self.assertEqual(fake_sentry.init_kwargs["max_breadcrumbs"], 150)
        self.assertEqual(len(fake_sentry.init_kwargs["integrations"]), 2)

    def test_init_sentry_configures_info_logs_as_breadcrumbs_only(self):
        from app.core.observability import init_sentry

        fake_sentry = FakeSentrySdk()
        config = SimpleNamespace(
            sentry_dsn="https://public@example.ingest.sentry.io/123",
            sentry_environment="develop",
            sentry_traces_sample_rate=0.0,
            sentry_max_breadcrumbs=100,
        )

        init_sentry(
            config,
            sentry_sdk_module=fake_sentry,
            fastapi_integration_cls=FakeFastApiIntegration,
            logging_integration_cls=FakeLoggingIntegration,
        )

        logging_integration = fake_sentry.init_kwargs["integrations"][1]
        self.assertEqual(logging_integration.kwargs["level"], logging.INFO)
        self.assertIsNone(logging_integration.kwargs["event_level"])

    def test_capture_exception_delegates_to_sentry_sdk(self):
        from app.core.observability import capture_exception

        fake_sentry = FakeSentrySdk()
        exc = RuntimeError("boom")

        captured = capture_exception(exc, sentry_sdk_module=fake_sentry)

        self.assertTrue(captured)
        self.assertEqual(fake_sentry.captured_exceptions, [exc])


if __name__ == "__main__":
    unittest.main()
