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
            GuideChatResponse,
            NextQuestionResponse,
            NextQuestionTurnClassification,
            TurnFeedbackResponse,
        )

        self.client = TestClient(app)
        self.conversation_route = conversation
        self.original_next_question = conversation.generate_next_question
        self.original_feedback = conversation.generate_feedback
        self.original_feedback_stream_events = getattr(conversation, "generate_feedback_stream_events", None)
        self.original_guide_answer = getattr(conversation, "generate_guide_answer", None)

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
        conversation.generate_feedback_stream_events = lambda request: iter([
            ("summary", {
                "comprehensionScore": 82,
                "feedbackSummary": "전체적으로 의도는 잘 전달됐지만 주문 표현이 조금 짧게 들립니다.",
            }),
            ("turnFeedback", {
                "turnId": 101,
                "feedbackRequired": True,
                "nativeUnderstanding": "외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.",
                "nativeLanguageInterpretation": "한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.",
                "betterExpression": "I'd like an iced Americano, please.",
            }),
            ("done", {"turnCount": 1}),
        ])
        conversation.generate_guide_answer = lambda request: GuideChatResponse(
            answer="would는 더 공손하고 부드러운 요청을 만들 때 자주 써요."
        )

    def tearDown(self):
        self.conversation_route.generate_next_question = self.original_next_question
        self.conversation_route.generate_feedback = self.original_feedback
        if self.original_feedback_stream_events is None:
            delattr(self.conversation_route, "generate_feedback_stream_events")
        else:
            self.conversation_route.generate_feedback_stream_events = self.original_feedback_stream_events
        if self.original_guide_answer is None:
            delattr(self.conversation_route, "generate_guide_answer")
        else:
            self.conversation_route.generate_guide_answer = self.original_guide_answer

    def test_next_question_route_returns_documented_shape(self):
        response = self.client.post("/api/v1/conversation/next-question", json={
            "originalQuestion": "What would you like to order?",
            "userUtterance": "I want iced americano.",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioSituation": "사용자는 주어진 시나리오 상황에서 상대방과 영어로 대화한다.",
            "aiRole": "상대방 역할",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "description": "테스트 슬롯 채움 기준", "filled": False},
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
            "scenarioSituation": "사용자는 주어진 시나리오 상황에서 상대방과 영어로 대화한다.",
            "aiRole": "상대방 역할",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "slots": [
                {"slotName": "drink", "description": "테스트 슬롯 채움 기준", "filled": True},
            ],
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

    def test_feedback_stream_route_returns_sse_events_in_order(self):
        with self.client.stream("POST", "/api/v1/conversation/feedback/stream", json={
            "scenarioTitle": "카페에서 주문하기",
            "scenarioSituation": "사용자는 주어진 시나리오 상황에서 상대방과 영어로 대화한다.",
            "aiRole": "상대방 역할",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "slots": [
                {"slotName": "drink", "description": "테스트 슬롯 채움 기준", "filled": True},
            ],
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                }
            ],
        }) as response:
            body = response.read().decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        self.assertLess(body.index("event: summary"), body.index("event: turnFeedback"))
        self.assertLess(body.index("event: turnFeedback"), body.index("event: done"))
        self.assertIn('"comprehensionScore":82', body)
        self.assertIn('"turnId":101', body)

    def test_feedback_stream_route_returns_error_event_when_generation_fails(self):
        def fail_stream(request):
            raise self.conversation_route.ConversationGenerationError("model unavailable")

        self.conversation_route.generate_feedback_stream_events = fail_stream

        with self.client.stream("POST", "/api/v1/conversation/feedback/stream", json={
            "scenarioTitle": "카페에서 주문하기",
            "scenarioSituation": "사용자는 주어진 시나리오 상황에서 상대방과 영어로 대화한다.",
            "aiRole": "상대방 역할",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "slots": [
                {"slotName": "drink", "description": "테스트 슬롯 채움 기준", "filled": True},
            ],
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                }
            ],
        }) as response:
            body = response.read().decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", body)
        self.assertIn('"code":"AI_GENERATION_FAILED"', body)

    def test_guide_route_returns_documented_shape(self):
        response = self.client.post("/api/v1/conversation/guide", json={
            "question": "would는 왜 쓰나요?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioSituation": "사용자는 카페에서 영어로 음료를 주문하는 상황입니다.",
            "aiRole": "카페 직원",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "originalQuestion": "What would you like to order?",
            "userUtterance": "I would like coffee.",
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": "would는 더 공손하고 부드러운 요청을 만들 때 자주 써요.",
        })

    def test_invalid_request_returns_documented_error_shape(self):
        response = self.client.post("/api/v1/conversation/feedback", json={
            "scenarioTitle": "카페에서 주문하기",
            "scenarioSituation": "사용자는 주어진 시나리오 상황에서 상대방과 영어로 대화한다.",
            "aiRole": "상대방 역할",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "slots": [
                {"slotName": "drink", "description": "테스트 슬롯 채움 기준", "filled": True},
            ],
            "turns": [],
        })

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {
            "code": "INVALID_REQUEST",
            "message": "잘못된 요청입니다.",
        })

    def test_missing_ai_role_returns_documented_error_shape(self):
        response = self.client.post("/api/v1/conversation/feedback", json={
            "scenarioTitle": "카페에서 주문하기",
            "scenarioSituation": "사용자는 주어진 시나리오 상황에서 상대방과 영어로 대화한다.",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "slots": [
                {"slotName": "drink", "description": "테스트 슬롯 채움 기준", "filled": True},
            ],
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                }
            ],
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
