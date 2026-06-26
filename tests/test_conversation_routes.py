# 3차 MVP 대화 API 라우팅 계약을 검증한다.
import os
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class ConversationRoutesTest(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api.routes import conversation
        from app.models.conversation import (
            ClosingMessageResponse,
            FeedbackType,
            GuideChatResponse,
            NextQuestionResponse,
            SessionFeedbackResponse,
            TurnFeedbackCreationResponse,
            TurnFeedbackData,
            TurnFeedbackStatus,
        )

        self.client = TestClient(app)
        self.conversation_route = conversation
        self.original_next_question = conversation.generate_next_question
        self.original_closing_message = conversation.generate_closing_message
        self.original_turn_feedback = conversation.generate_turn_feedback
        self.original_session_feedback = conversation.generate_session_feedback
        self.original_guide_answer = conversation.generate_guide_answer
        self.original_capture_exception = conversation.capture_exception

        conversation.generate_next_question = lambda request: NextQuestionResponse(
            aiQuestion="Oh, you like spicy pizza. Do you cook often?",
            translatedQuestion="매운 피자를 좋아하는군요. 요리는 자주 하나요?",
            innerThought="매운 피자를 좋아한다고 이유까지 말해주니 대화가 편하네요.",
            innerThoughtType="GOOD",
        )
        conversation.generate_closing_message = lambda request: ClosingMessageResponse(
            aiMessage="Got it. Let's wrap up here.",
            translatedMessage="알겠어. 여기서 마무리하자.",
            innerThought="마지막 답변까지 들었으니 자연스럽게 마무리해도 되겠다.",
            innerThoughtType="GOOD",
        )
        conversation.generate_turn_feedback = lambda request: TurnFeedbackCreationResponse(
            sessionId=request.sessionId,
            turnId=request.turnId,
            feedbackStatus=TurnFeedbackStatus.PREPARING,
        )
        conversation.generate_session_feedback = lambda request: SessionFeedbackResponse(
            sessionId=request.sessionId,
            nativeScore=82,
            highlightMessage="한국인의 35%가 틀리는 이유 연결을 정확히 맞춘 사람",
            turnFeedbacks=[
                TurnFeedbackData(
                    turnId=5000,
                    feedbackType=FeedbackType.GOOD,
                    koreanAnalogy="한국어로 비유하자면 '저는 피자가 좋아요. 매워서요'처럼 담백하게 들려요.",
                    positiveFeedback=None,
                    feedbackDetail="이유를 because로 자연스럽게 붙였고, 좋아하는 음식과 이유를 한 문장 안에서 분명하게 연결했어요.",
                    benchmarkMessage="한국인의 35%가 틀리는 이유 연결을 정확히 맞춘 사람",
                )
            ],
        )
        conversation.generate_guide_answer = lambda request: GuideChatResponse(
            answer="would는 더 공손하고 부드러운 요청을 만들 때 자주 써요."
        )

    def tearDown(self):
        self.conversation_route.generate_next_question = self.original_next_question
        self.conversation_route.generate_closing_message = self.original_closing_message
        self.conversation_route.generate_turn_feedback = self.original_turn_feedback
        self.conversation_route.generate_session_feedback = self.original_session_feedback
        self.conversation_route.generate_guide_answer = self.original_guide_answer
        self.conversation_route.capture_exception = self.original_capture_exception

    def _next_question_payload(self):
        return {
            "sessionId": 1000,
            "submittedTurnId": 5000,
            "submittedSequence": 1,
            "scenario": {
                "scenarioId": 10,
                "title": "음식에 대한 대화하기",
                "briefing": "좋아하는 음식과 최근 먹었던 음식에 대해 이야기합니다.",
                "conversationGoal": "음식 취향과 경험을 영어로 자연스럽게 설명할 수 있다.",
                "counterpartRole": "friend",
            },
            "currentTurn": {
                "aiQuestion": "What is your favorite food? Why do you like it?",
                "translatedQuestion": "가장 좋아하는 음식이 뭐예요? 왜 좋아하나요?",
                "userUtterance": "I like pizza because it is spicy.",
            },
            "nextQuestion": {
                "questionId": 101,
                "sequence": 2,
                "questionEn": "Do you cook often?",
                "questionKo": "요리는 자주 하나요?",
            },
        }

    def _turn_feedback_payload(self, *, turn_id=5000):
        return {
            "sessionId": 1000,
            "turnId": turn_id,
            "sequence": 1,
            "scenario": {
                "scenarioId": 10,
                "title": "음식에 대한 대화하기",
                "briefing": "좋아하는 음식과 최근 먹었던 음식에 대해 이야기합니다.",
                "conversationGoal": "음식 취향과 경험을 영어로 자연스럽게 설명할 수 있다.",
                "counterpartRole": "friend",
            },
            "turn": {
                "aiQuestion": "What is your favorite food? Why do you like it?",
                "translatedQuestion": "가장 좋아하는 음식이 뭐예요? 왜 좋아하나요?",
                "userUtterance": "I like pizza because it is spicy.",
            },
        }

    def test_next_question_route_returns_documented_shape(self):
        response = self.client.post("/api/v1/conversation/next-question", json=self._next_question_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "aiQuestion": "Oh, you like spicy pizza. Do you cook often?",
            "translatedQuestion": "매운 피자를 좋아하는군요. 요리는 자주 하나요?",
            "innerThought": "매운 피자를 좋아한다고 이유까지 말해주니 대화가 편하네요.",
            "innerThoughtType": "GOOD",
        })

    def test_next_question_route_propagates_request_id_to_context_and_response(self):
        seen_request_ids = []

        def record_request_id(request):
            from app.core.request_context import get_request_id
            from app.models.conversation import NextQuestionResponse

            seen_request_ids.append(get_request_id())
            return NextQuestionResponse(
                aiQuestion="Oh, you like spicy pizza. Do you cook often?",
                translatedQuestion="매운 피자를 좋아하는군요. 요리는 자주 하나요?",
                innerThought="매운 피자를 좋아한다고 이유까지 말해주니 대화가 편하네요.",
                innerThoughtType="GOOD",
            )

        self.conversation_route.generate_next_question = record_request_id

        response = self.client.post(
            "/api/v1/conversation/next-question",
            headers={"X-Request-Id": "trace-ai-3mvp"},
            json=self._next_question_payload(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-Id"], "trace-ai-3mvp")
        self.assertEqual(seen_request_ids, ["trace-ai-3mvp"])

    def test_closing_message_route_returns_documented_shape(self):
        payload = self._next_question_payload()
        payload.pop("nextQuestion")
        payload["closingReason"] = "GOAL_COMPLETED"
        payload["goalCompletionStatus"] = "COMPLETED"

        response = self.client.post("/api/v1/conversation/closing-message", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "aiMessage": "Got it. Let's wrap up here.",
            "translatedMessage": "알겠어. 여기서 마무리하자.",
            "innerThought": "마지막 답변까지 들었으니 자연스럽게 마무리해도 되겠다.",
            "innerThoughtType": "GOOD",
        })

    def test_turn_feedback_route_returns_preparing_shape(self):
        response = self.client.post("/api/v1/conversation/turn-feedback", json=self._turn_feedback_payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "sessionId": 1000,
            "turnId": 5000,
            "feedbackStatus": "PREPARING",
        })

    def test_session_feedback_route_returns_documented_shape(self):
        response = self.client.post("/api/v1/conversation/session-feedback", json={
            "sessionId": 1000,
            "scenario": {
                "scenarioId": 10,
                "title": "음식에 대한 대화하기",
                "briefing": "좋아하는 음식과 최근 먹었던 음식에 대해 이야기합니다.",
                "conversationGoal": "음식 취향과 경험을 영어로 자연스럽게 설명할 수 있다.",
                "counterpartRole": "friend",
            },
            "expectedTurnIds": [5000],
        })

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sessionId"], 1000)
        self.assertEqual(body["nativeScore"], 82)
        self.assertEqual(body["highlightMessage"], "한국인의 35%가 틀리는 이유 연결을 정확히 맞춘 사람")
        self.assertNotIn("nativeScoreBreakdown", body)
        self.assertNotIn("nativeLevelLabel", body)
        self.assertNotIn("summary", body)
        self.assertEqual(body["turnFeedbacks"][0]["feedbackType"], "GOOD")
        self.assertIn("feedbackDetail", body["turnFeedbacks"][0])
        self.assertNotIn("betterExpression", body["turnFeedbacks"][0])
        self.assertEqual(
            body["turnFeedbacks"][0]["benchmarkMessage"],
            "한국인의 35%가 틀리는 이유 연결을 정확히 맞춘 사람",
        )

    def test_session_feedback_route_returns_not_ready_error(self):
        captured = []

        def not_ready(request):
            raise self.conversation_route.TurnFeedbackNotReadyError([5001])

        self.conversation_route.generate_session_feedback = not_ready
        self.conversation_route.capture_exception = lambda exc: captured.append(exc)

        response = self.client.post("/api/v1/conversation/session-feedback", json={
            "sessionId": 1000,
            "scenario": {
                "scenarioId": 10,
                "title": "음식에 대한 대화하기",
                "briefing": "좋아하는 음식과 최근 먹었던 음식에 대해 이야기합니다.",
                "conversationGoal": "음식 취향과 경험을 영어로 자연스럽게 설명할 수 있다.",
                "counterpartRole": "friend",
            },
            "expectedTurnIds": [5000, 5001],
        })

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "TURN_FEEDBACK_NOT_READY")
        self.assertEqual(captured, [])

    def test_generation_error_returns_documented_error(self):
        captured = []

        def fail_next_question(request):
            raise self.conversation_route.ConversationGenerationError("model unavailable")

        self.conversation_route.generate_next_question = fail_next_question
        self.conversation_route.capture_exception = lambda exc: captured.append(exc)

        response = self.client.post("/api/v1/conversation/next-question", json=self._next_question_payload())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["code"], "AI_GENERATION_FAILED")
        self.assertEqual(len(captured), 1)

    def test_legacy_feedback_routes_are_not_registered(self):
        self.assertEqual(self.client.post("/api/v1/conversation/feedback", json={}).status_code, 404)
        self.assertEqual(self.client.post("/api/v1/conversation/feedback/stream", json={}).status_code, 404)

    def test_guide_route_stays_available(self):
        response = self.client.post("/api/v1/conversation/guide", json={
            "question": "I would like coffee에서 would는 왜 쓰나요?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioSituation": "사용자는 카페에서 영어로 음료를 주문하는 상황입니다.",
            "aiRole": "카페 직원",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": "would는 더 공손하고 부드러운 요청을 만들 때 자주 써요.",
        })

    def test_invalid_request_returns_documented_error_shape(self):
        payload = self._turn_feedback_payload()
        payload["turn"]["userUtterance"] = "   "

        response = self.client.post("/api/v1/conversation/turn-feedback", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {
            "code": "INVALID_REQUEST",
            "message": "잘못된 요청입니다.",
        })


if __name__ == "__main__":
    unittest.main()
