# 세션 피드백 서비스가 백엔드 문서 계약 payload를 처리하는지 검증하는 테스트
import importlib
import json
import os
import unittest


class SessionFeedbackServiceTest(unittest.TestCase):

    def test_build_feedback_accepts_backend_contract_payload(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        models = importlib.import_module("app.models.session_feedback")
        service = importlib.import_module("app.services.session_feedback_service")

        payload = {
            "sessionId": "6db9bf56-11ba-4fab-8f50-1e4c3016d82d",
            "scenario": {
                "scenarioId": "cafe_iced_americano",
                "title": "아이스 아메리카노 주문하기",
                "situationDescription": "카페에서 원하는 음료를 주문해야 합니다.",
                "successGoal": "아이스 아메리카노 주문에 성공하세요.",
            },
            "scenarioResult": "SUCCESS",
            "filledSlots": [
                {"slotKey": "drink", "slotValue": "americano"},
                {"slotKey": "temperature", "slotValue": "iced"},
            ],
            "turns": [
                {
                    "turnId": 1,
                    "turnIndex": 1,
                    "questionText": "Hi! What would you like to order?",
                    "userTranscript": "I want iced americano",
                    "speechStartedAfterMs": 2100,
                    "recordingDurationMs": 3600,
                },
                {
                    "turnId": 2,
                    "turnIndex": 2,
                    "questionText": "What size would you like?",
                    "userTranscript": "Small, please.",
                    "speechStartedAfterMs": 1200,
                    "recordingDurationMs": 1600,
                },
            ],
        }

        responses = iter([
            json.dumps({
                "comprehension_score": 88,
                "native_perception": "The speaker wants an iced Americano.",
                "better_expression": "I'd like an iced Americano, please.",
            }),
            json.dumps({"score": 94}),
            "주문 의도는 잘 전달됐고 더 정중한 표현을 쓰면 좋습니다.",
            json.dumps({
                "comprehension_score": 92,
                "native_perception": "The speaker wants a small size.",
                "better_expression": "Small, please.",
            }),
            json.dumps({"score": 95}),
            "간결하고 자연스럽게 답했습니다.",
            "전체적으로 주문 목표를 잘 달성했습니다.",
        ])
        service.chat = lambda *args, **kwargs: next(responses)

        request = models.SessionFeedbackRequest.model_validate(payload)
        result = service.build_feedback(request)

        self.assertEqual(result.totalUnderstoodScore, 90)
        self.assertEqual(result.summary, "전체적으로 주문 목표를 잘 달성했습니다.")
        self.assertEqual(len(result.turns), 2)
        self.assertEqual(result.turns[0].heardAs, "The speaker wants an iced Americano.")
        self.assertEqual(result.turns[1].scoreDelta, 4)


if __name__ == "__main__":
    unittest.main()
