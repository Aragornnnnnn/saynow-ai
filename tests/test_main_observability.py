# FastAPI 전역 예외 처리의 관측성 연결을 검증하는 테스트
import asyncio
import os
from types import SimpleNamespace
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class MainObservabilityTest(unittest.TestCase):

    def test_internal_exception_handler_captures_exception(self):
        from app import main as main_module

        captured = []
        original_capture_exception = main_module.capture_exception
        main_module.capture_exception = lambda exc: captured.append(exc)
        request = SimpleNamespace(url=SimpleNamespace(path="/boom"))
        exc = RuntimeError("boom")

        try:
            response = asyncio.run(main_module.internal_exception_handler(request, exc))
        finally:
            main_module.capture_exception = original_capture_exception

        self.assertEqual(response.status_code, 500)
        self.assertEqual(captured, [exc])


if __name__ == "__main__":
    unittest.main()
