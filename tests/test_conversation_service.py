# 2차 MVP 대화 API 서비스 계약을 검증하는 테스트
import json
import os
import unittest


os.environ.setdefault("OPENAI_API_KEY", "test-key")


class ConversationServiceTest(unittest.TestCase):

    def setUp(self):
        from app.services import conversation_service

        self.service = conversation_service
        self.original_chat = conversation_service.chat

    def tearDown(self):
        self.service.chat = self.original_chat

    def test_next_question_returns_only_newly_filled_unfilled_slots(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "I want an iced americano.",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": True},
                {"slotName": "size", "filled": False},
                {"slotName": "temperature", "filled": False},
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "filledSlots": [
                {"slotName": "drink"},
                {"slotName": "temperature"},
            ],
            "nextQuestion": "What size would you like?",
            "translatedQuestion": "어떤 사이즈로 드릴까요?",
        })

        result = self.service.generate_next_question(request)

        self.assertEqual([slot.slotName for slot in result.filledSlots], ["temperature"])
        self.assertEqual(result.nextQuestion, "What size would you like?")
        self.assertEqual(result.translatedQuestion, "어떤 사이즈로 드릴까요?")

    def test_next_question_returns_null_when_all_unfilled_slots_are_newly_filled(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "Small iced americano, please.",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": True},
                {"slotName": "size", "filled": False},
                {"slotName": "temperature", "filled": False},
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "filledSlots": [
                {"slotName": "size"},
                {"slotName": "temperature"},
            ],
            "nextQuestion": "Ignored question",
            "translatedQuestion": "무시되는 질문",
        })

        result = self.service.generate_next_question(request)

        self.assertEqual([slot.slotName for slot in result.filledSlots], ["size", "temperature"])
        self.assertIsNone(result.nextQuestion)
        self.assertIsNone(result.translatedQuestion)

    def test_feedback_preserves_backend_turn_ids_and_feedback_fields(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
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
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 82,
            "feedbackSummary": "전체적으로 의도는 잘 전달됐지만 주문 표현이 조금 짧게 들립니다.",
            "turnFeedbacks": [
                {
                    "turnId": 101,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "아이스 아메리카노를 주문하고 싶다는 의미로 이해됩니다.",
                    "nativeLanguageInterpretation": "나 아이스 아메리카노 원해처럼 조금 직접적으로 들립니다.",
                    "betterExpression": "I'd like an iced Americano, please.",
                }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertEqual(result.comprehensionScore, 82)
        self.assertEqual(result.turnFeedbacks[0].turnId, 101)
        self.assertTrue(result.turnFeedbacks[0].feedbackRequired)
        self.assertEqual(result.turnFeedbacks[0].betterExpression, "I'd like an iced Americano, please.")

    def test_feedback_invalid_model_json_raises_generation_error(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
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
        self.service.chat = lambda *args, **kwargs: "not json"

        with self.assertRaises(self.service.ConversationGenerationError):
            self.service.generate_feedback(request)

    def test_feedback_prompt_contains_stable_good_response_rubric_and_plus_one_policy(self):
        prompt = self.service._feedback_system_prompt()

        self.assertIn("Stable feedback decision rubric", prompt)
        self.assertIn("85-100", prompt)
        self.assertIn("feedbackRequired=false", prompt)
        self.assertIn("Only set feedbackRequired=false when all Good Response Conditions pass", prompt)
        self.assertIn("betterExpression +1 policy", prompt)
        self.assertIn("Keep the user's original intent, vocabulary level, and sentence shape", prompt)

    def test_feedback_prompt_constrains_turn_feedback_copy_contract(self):
        prompt = self.service._feedback_system_prompt()

        self.assertIn("nativeUnderstanding must explain what the foreign listener understood", prompt)
        self.assertIn("외국인은", prompt)
        self.assertIn("라고 이해했어요", prompt)
        self.assertIn("nativeLanguageInterpretation must be a Korean analogy", prompt)
        self.assertIn("한국어로 비유하자면", prompt)
        self.assertIn("betterExpression must include the improved sentence and a short Korean reason", prompt)

    def test_feedback_uses_deterministic_chat_settings(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
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
        captured = {}

        def capture_chat(*args, **kwargs):
            captured.update(kwargs)
            return json.dumps({
                "comprehensionScore": 82,
                "feedbackSummary": "전체적으로 의도는 전달됐지만 표현을 조금 다듬으면 좋습니다.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "아이스 아메리카노를 주문하고 싶다는 의미로 이해됩니다.",
                        "nativeLanguageInterpretation": "나 아이스 아메리카노 원해처럼 조금 직접적으로 들립니다.",
                        "betterExpression": "I'd like an iced Americano, please.",
                    }
                ],
            })

        self.service.chat = capture_chat

        self.service.generate_feedback(request)

        self.assertEqual(captured["temperature"], 0)

    def test_next_question_model_call_failure_raises_generation_error(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "I want iced americano.",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
            ],
        })

        def fail_chat(*args, **kwargs):
            raise RuntimeError("model unavailable")

        self.service.chat = fail_chat

        with self.assertRaises(self.service.ConversationGenerationError):
            self.service.generate_next_question(request)


if __name__ == "__main__":
    unittest.main()
