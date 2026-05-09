# STT 신뢰도 계산에서 OpenAI verbose_json segment 구조를 검증하는 테스트
import importlib
import math
import sys
import types
import unittest
from types import SimpleNamespace


class _FakeOpenAI:
    def __init__(self, api_key):
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(text="", segments=[])
            )
        )


def _load_stt_service():
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI

    fake_config = types.ModuleType("app.config")
    fake_config.settings = SimpleNamespace(openai_api_key="test-key")

    sys.modules["openai"] = fake_openai
    sys.modules["app.config"] = fake_config
    sys.modules.pop("app.services.stt_service", None)
    return importlib.import_module("app.services.stt_service")


class SttServiceTest(unittest.TestCase):

    def test_transcribe_with_confidence_accepts_dict_segments(self):
        stt_service = _load_stt_service()
        response = SimpleNamespace(
            text="I want an iced americano.",
            segments=[
                {"avg_logprob": -0.1},
                {"avg_logprob": -0.3},
            ],
        )
        stt_service._client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )

        result = stt_service.transcribe_with_confidence(b"audio", "turn.mp3")

        self.assertEqual(result["text"], "I want an iced americano.")
        self.assertEqual(result["confidence"], round(math.exp(-0.2), 4))


if __name__ == "__main__":
    unittest.main()
