# 백엔드가 전달한 시나리오 payload만으로 턴 평가가 가능한지 검증하는 테스트
import importlib
import sys
import types
import unittest


class _FilledSlot:
    def __init__(self, slotKey, slotValue):
        self.slotKey = slotKey
        self.slotValue = slotValue


class _TtsContent:
    def __init__(self, questionText=None, messageText=None, ttsAudio=""):
        self.questionText = questionText
        self.messageText = messageText
        self.ttsAudio = ttsAudio


class _TurnEvaluationResponse:
    def __init__(
        self,
        transcript,
        sttConfidence,
        scenarioStatus,
        filledSlots,
        nextQuestion=None,
        resultMessage=None,
    ):
        self.transcript = transcript
        self.sttConfidence = sttConfidence
        self.scenarioStatus = scenarioStatus
        self.filledSlots = filledSlots
        self.nextQuestion = nextQuestion
        self.resultMessage = resultMessage


def _load_turn_evaluation_service():
    fake_models = types.ModuleType("app.models.turn_evaluation")
    fake_models.FilledSlot = _FilledSlot
    fake_models.TtsContent = _TtsContent
    fake_models.TurnEvaluationResponse = _TurnEvaluationResponse

    fake_llm = types.ModuleType("app.core.llm")
    fake_llm.chat = lambda *args, **kwargs: "What size would you like?"

    fake_stt = types.ModuleType("app.services.stt_service")
    fake_stt.transcribe_with_confidence = lambda *args, **kwargs: {
        "text": "I want an iced Americano.",
        "confidence": 0.8,
    }

    fake_tts = types.ModuleType("app.services.tts_service")
    fake_tts.synthesize = lambda *args, **kwargs: "audio"

    fake_scenario = types.ModuleType("app.services.scenario_service")
    fake_scenario.get_by_id = lambda scenario_id: None

    sys.modules["app.models.turn_evaluation"] = fake_models
    sys.modules["app.core.llm"] = fake_llm
    sys.modules["app.services.stt_service"] = fake_stt
    sys.modules["app.services.tts_service"] = fake_tts
    sys.modules["app.services.scenario_service"] = fake_scenario
    sys.modules.pop("app.services.turn_evaluation_service", None)
    return importlib.import_module("app.services.turn_evaluation_service")


class TurnEvaluationServiceTest(unittest.TestCase):

    def test_evaluate_turn_uses_request_payload_without_internal_scenario_lookup(self):
        service = _load_turn_evaluation_service()
        service._extract_slots = lambda *args, **kwargs: [_FilledSlot("drink", "iced Americano")]
        service._generate_followup = lambda *args, **kwargs: "What size would you like?"

        result = service.evaluate_turn(
            audio_bytes=b"audio",
            filename="turn.mp3",
            scenario_id="cafe_iced_americano",
            scenario_situation="카페에서 원하는 음료를 주문해야 합니다.",
            scenario_goal="아이스 아메리카노 주문에 성공하세요.",
            required_keys=["drink", "size"],
            max_follow_up_count=5,
            current_question="Hi! What would you like to order?",
            filled_slots=[],
            conversation_history=[],
        )

        self.assertEqual(result.scenarioStatus, "IN_PROGRESS")
        self.assertEqual(result.filledSlots[0].slotKey, "drink")
        self.assertEqual(result.nextQuestion.questionText, "What size would you like?")


if __name__ == "__main__":
    unittest.main()
