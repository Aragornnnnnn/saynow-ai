# 2차 MVP 대화 API 라우팅 계약을 검증하는 테스트
import os
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class ConversationRoutesTest(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api.routes import conversation
        from app.models.conversation import (
            ConversationFeedbackResponse,
            FilledSlotResponse,
            NextQuestionResponse,
            NextQuestionTurnClassification,
            TurnFeedbackResponse,
        )

        self.client = TestClient(app)
        self.conversation_route = conversation
        self.original_next_question = conversation.generate_next_question
        self.original_feedback = conversation.generate_feedback

        conversation.generate_next_question = lambda request: NextQuestionResponse(
            nextQuestion="What size would you like?",
            translatedQuestion="어떤 사이즈로 드릴까요?",
            filledSlots=[FilledSlotResponse(slotName="drink")],
            turnClassification=NextQuestionTurnClassification.ANSWER,
        )
        conversation.generate_feedback = lambda request: ConversationFeedbackResponse(
            comprehensionScore=82,
            feedbackSummary="전체적으로 의도는 잘 전달됐지만 주문 표현이 조금 짧게 들립니다.",
            turnFeedbacks=[
                TurnFeedbackResponse(
                    turnId=101,
                    feedbackRequired=True,
                    nativeUnderstanding="아이스 아메리카노를 주문하고 싶다는 의미로 이해됩니다.",
                    nativeLanguageInterpretation="나 아이스 아메리카노 원해처럼 조금 직접적으로 들립니다.",
                    betterExpression="I'd like an iced Americano, please.",
                )
            ],
        )

    def tearDown(self):
        self.conversation_route.generate_next_question = self.original_next_question
        self.conversation_route.generate_feedback = self.original_feedback

    def test_next_question_route_returns_documented_shape(self):
        response = self.client.post("/api/v1/conversation/next-question", json={
            "originalQuestion": "What would you like to order?",
            "userUtterance": "I want iced americano.",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
            ],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "nextQuestion": "What size would you like?",
            "translatedQuestion": "어떤 사이즈로 드릴까요?",
            "filledSlots": [{"slotName": "drink"}],
            "turnClassification": "ANSWER",
        })

    def test_feedback_route_returns_documented_shape(self):
        response = self.client.post("/api/v1/conversation/feedback", json={
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                }
            ],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["comprehensionScore"], 82)
        self.assertEqual(response.json()["turnFeedbacks"][0]["turnId"], 101)
        self.assertTrue(response.json()["turnFeedbacks"][0]["feedbackRequired"])

    def test_invalid_request_returns_documented_error_shape(self):
        response = self.client.post("/api/v1/conversation/feedback", json={
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "turns": [],
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {
            "code": "INVALID_REQUEST",
            "message": "잘못된 요청입니다.",
        })

    def test_old_turn_evaluation_endpoint_is_not_registered(self):
        response = self.client.post("/api/v1/turn-evaluations")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
