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

    def test_next_question_blocks_non_answer_utterances_even_when_model_returns_slots(self):
        from app.models.conversation import NextQuestionRequest

        blocked_utterances = [
            "qwertyuiop asdfghjkl zxcvbnm",
            "My shoes are swimming in the moon today.",
            "I don't know.",
            "No answer.",
            "I do not want to order anything.",
        ]

        for utterance in blocked_utterances:
            with self.subTest(utterance=utterance):
                request = NextQuestionRequest.model_validate({
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": utterance,
                    "scenarioTitle": "카페에서 주문하기",
                    "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
                    "slots": [
                        {"slotName": "drink", "filled": False},
                        {"slotName": "size", "filled": False},
                    ],
                })
                self.service.chat = lambda *args, **kwargs: json.dumps({
                    "filledSlots": [
                        {"slotName": "drink"},
                        {"slotName": "size"},
                    ],
                    "nextQuestion": None,
                    "translatedQuestion": None,
                })

                result = self.service.generate_next_question(request)

                self.assertEqual(result.filledSlots, [])
                self.assertEqual(result.nextQuestion, "What drink would you like to order?")
                self.assertEqual(result.translatedQuestion, "어떤 음료를 주문하고 싶으신가요?")

    def test_next_question_prompt_requires_explicit_slot_evidence(self):
        prompt = self.service._next_question_system_prompt()

        self.assertIn("Only mark a slot as filled when the user explicitly provides a concrete value", prompt)
        self.assertIn("Nonsense, off-topic, refusal, or vague non-answer utterances must return filledSlots=[]", prompt)
        self.assertIn("qwertyuiop asdfghjkl zxcvbnm", prompt)
        self.assertIn("My shoes are swimming in the moon today", prompt)
        self.assertIn("I don't know", prompt)
        self.assertIn("I do not want to order anything", prompt)

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

    def test_feedback_caps_non_answer_score_even_when_model_scores_high(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I don't know.",
                }
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 82,
            "feedbackSummary": "주문 의도를 명확히 전달하지 못해 자연스러운 주문으로 이어지지 않았습니다.",
            "turnFeedbacks": [
                {
                    "turnId": 101,
                    "feedbackRequired": False,
                    "nativeUnderstanding": None,
                    "nativeLanguageInterpretation": None,
                    "betterExpression": None,
                }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertEqual(result.comprehensionScore, 39)
        self.assertTrue(result.turnFeedbacks[0].feedbackRequired)
        self.assertTrue(result.turnFeedbacks[0].betterExpression.startswith("I'd like a coffee, please."))

    def test_feedback_rewrites_leaked_native_language_examples_for_cafe_option_turns(self):
        from app.models.conversation import ConversationFeedbackRequest

        cases = [
            (
                "I want ice one.",
                "한국어로 비유하자면, '아침식사 몇 시'처럼 들려요.",
                "한국어로 비유하자면, '얼음 하나 원해요'처럼 들려요.",
            ),
            (
                "Less ice do please.",
                "한국어로 비유하자면, '목성 날씨가 파란 삼각형 맛이 난다'처럼 들려요.",
                "한국어로 비유하자면, '얼음 적게 해주세요'처럼 들려요.",
            ),
            (
                "This drink is hot but I order ice one.",
                "한국어로 비유하자면, '이 음료는 뜨겁지만 얼음 한 개를 주문했어요'처럼 들려요.",
                "한국어로 비유하자면, '이 음료는 뜨겁지만 얼음 한 개를 주문했어요'처럼 들려요.",
            ),
        ]

        for user_utterance, model_interpretation, expected_interpretation in cases:
            with self.subTest(user_utterance=user_utterance):
                request = ConversationFeedbackRequest.model_validate({
                    "scenarioTitle": "카페에서 옵션 말하기",
                    "scenarioGoal": "음료 옵션을 자연스럽게 말할 수 있다.",
                    "turns": [
                        {
                            "turnId": 101,
                            "originalQuestion": "Would you like it hot or iced?",
                            "userUtterance": user_utterance,
                        }
                    ],
                })
                self.service.chat = lambda *args, **kwargs: json.dumps({
                    "comprehensionScore": 72,
                    "feedbackSummary": "의미는 일부 전달됐지만 옵션 표현을 더 명확히 다듬으면 좋습니다.",
                    "turnFeedbacks": [
                        {
                            "turnId": 101,
                            "feedbackRequired": True,
                            "nativeUnderstanding": "외국인은 사용자가 얼음이나 차가운 옵션을 원한다고 이해했어요.",
                            "nativeLanguageInterpretation": model_interpretation,
                            "betterExpression": "I'd like it iced, please. 이렇게 말하면 차가운 옵션을 더 명확하게 전달할 수 있어요.",
                        }
                    ],
                })

                result = self.service.generate_feedback(request)

                self.assertEqual(result.turnFeedbacks[0].nativeLanguageInterpretation, expected_interpretation)

    def test_feedback_rewrites_native_understanding_to_required_format_for_known_cases(self):
        from app.models.conversation import ConversationFeedbackRequest

        cases = [
            (
                "I want ice one.",
                "사용자가 음료에 얼음을 넣고 싶다는 의미로 이해했습니다. 얼음을 한 개만 넣겠다는 뜻으로 들렸습니다.",
                "외국인은 사용자가 얼음 한 개를 원한다고 이해했어요.",
            ),
            (
                "Less ice do please.",
                "사용자가 얼음을 적게 넣어 달라는 의미로 이해했습니다. Less ice do please는 문법적으로 어색해 정확한 의도를 파악하기 어려웠습니다.",
                "외국인은 사용자가 얼음을 적게 넣어 달라고 이해했어요.",
            ),
            (
                "My shoes are swimming in the moon today.",
                "사용자가 신발을 주문하고 싶어한다고 들립니다. 신발이 달이나 물속에서 헤엄치는 상황을 상상하고 있는 것으로 보입니다.",
                "외국인은 사용자가 신발이 달에서 수영하고 있다고 말한다고 이해했어요.",
            ),
        ]

        for user_utterance, model_understanding, expected_understanding in cases:
            with self.subTest(user_utterance=user_utterance):
                request = ConversationFeedbackRequest.model_validate({
                    "scenarioTitle": "카페에서 옵션 말하기",
                    "scenarioGoal": "음료 옵션을 자연스럽게 말할 수 있다.",
                    "turns": [
                        {
                            "turnId": 101,
                            "originalQuestion": "Would you like it hot or iced?",
                            "userUtterance": user_utterance,
                        }
                    ],
                })
                self.service.chat = lambda *args, **kwargs: json.dumps({
                    "comprehensionScore": 72,
                    "feedbackSummary": "의미는 일부 전달됐지만 표현을 더 명확히 다듬으면 좋습니다.",
                    "turnFeedbacks": [
                        {
                            "turnId": 101,
                            "feedbackRequired": True,
                            "nativeUnderstanding": model_understanding,
                            "nativeLanguageInterpretation": "한국어로 비유하자면, '테스트 문장'처럼 들려요.",
                            "betterExpression": "I'd like it iced, please. 이렇게 말하면 차가운 옵션을 더 명확하게 전달할 수 있어요.",
                        }
                    ],
                })

                result = self.service.generate_feedback(request)
                native_understanding = result.turnFeedbacks[0].nativeUnderstanding

                self.assertEqual(native_understanding, expected_understanding)
                self.assertTrue(native_understanding.startswith("외국인은"))
                self.assertRegex(native_understanding, r"(라고|다고) 이해했어요\.$")
                self.assertNotIn("문법적으로", native_understanding)
                self.assertNotIn("정확한 의도를 파악하기 어려웠습니다", native_understanding)

    def test_feedback_rewrites_off_topic_native_language_interpretation_to_literal_meaning(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "My shoes are swimming in the moon today.",
                }
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 35,
            "feedbackSummary": "음료 주문 의도가 전달되지 않아 시나리오 목표를 달성하지 못했습니다.",
            "turnFeedbacks": [
                {
                    "turnId": 101,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "외국인은 신발이 달에서 수영한다는 이상한 설명으로 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '신발이 달에서 헤엄치는 것처럼 들려서 음료 주문과는 전혀 관련이 없어 보여요.'처럼 들려요.",
                    "betterExpression": "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 주문할 수 있어요.",
                }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertEqual(
            result.turnFeedbacks[0].nativeLanguageInterpretation,
            "한국어로 비유하자면, '달에서 신발이 수영한다'처럼 들려요.",
        )

    def test_feedback_prompt_contains_stable_good_response_rubric_and_plus_one_policy(self):
        prompt = self.service._feedback_system_prompt()

        self.assertIn("Stable feedback decision rubric", prompt)
        self.assertIn("85-100", prompt)
        self.assertIn("feedbackRequired=false", prompt)
        self.assertIn("Only set feedbackRequired=false when all Good Response Conditions pass", prompt)
        self.assertIn("betterExpression +1 policy", prompt)
        self.assertIn("Keep the user's original intent, vocabulary level, and sentence shape", prompt)
        self.assertIn("If the scenario goal is not achieved, comprehensionScore must be 59 or below", prompt)
        self.assertIn("Nonsense, off-topic, refusal, or vague non-answer utterances must score 0-39", prompt)
        self.assertIn("betterExpression must start with the English improved sentence", prompt)
        self.assertNotIn("음료를 주문할 때는 I'd like", prompt)
        self.assertIn("I want ice one", prompt)
        self.assertIn("I'd like it iced, please.", prompt)
        self.assertIn("This drink is hot, but I ordered an iced one.", prompt)

    def test_feedback_prompt_constrains_turn_feedback_copy_contract(self):
        prompt = self.service._feedback_system_prompt()

        self.assertIn("nativeUnderstanding must explain what the foreign listener understood", prompt)
        self.assertIn("nativeUnderstanding must start with '외국인은'", prompt)
        self.assertIn("nativeUnderstanding must end with '라고 이해했어요.'", prompt)
        self.assertIn("nativeUnderstanding must be based only on the same turn's userUtterance", prompt)
        self.assertIn("Do not include grammar explanations, improvement directions, or evaluations in nativeUnderstanding", prompt)
        self.assertIn("Do not quote the user's utterance in nativeUnderstanding", prompt)
        self.assertIn("describe the practical intent, uncertainty, or likely misunderstanding", prompt)
        self.assertIn("nativeLanguageInterpretation must be a Korean analogy", prompt)
        self.assertIn("한국어로 비유하자면", prompt)
        self.assertIn("nativeLanguageInterpretation must be based only on the same turn's userUtterance", prompt)
        self.assertIn("Do not borrow content from prompt examples, previous turns, other test inputs, scenarioTitle, or scenarioGoal", prompt)
        self.assertIn("nativeUnderstanding and nativeLanguageInterpretation must describe the same meaning", prompt)
        self.assertIn("Use single quotation marks around the Korean analogy phrase in nativeLanguageInterpretation", prompt)
        self.assertIn("betterExpression must include the improved sentence and a short Korean reason", prompt)
        self.assertIn("Do not include backslash characters", prompt)
        self.assertIn("Do not use double quotation marks inside any response string", prompt)
        self.assertNotIn("아침식사 몇 시", prompt)

    def test_feedback_prompt_constrains_off_topic_feedback_format(self):
        prompt = self.service._feedback_system_prompt()

        self.assertIn("For nonsensical or off-topic utterances", prompt)
        self.assertIn("preserve the strange meaning in the Korean analogy", prompt)
        self.assertIn("do not force it into the scenario context", prompt)
        self.assertIn("When the user's utterance does not answer the AI question or scenario intent", prompt)
        self.assertIn("give a simple English answer without wrapping it in quotation marks", prompt)
        self.assertIn("Do not return only an English sentence with a parenthesized Korean translation", prompt)
        self.assertIn("Do not write nativeUnderstanding as '주문할 음료에 대한 내용이 아니다'", prompt)
        self.assertIn("The English example must appear plainly without double quotation marks", prompt)
        self.assertIn("For nonsensical utterances, nativeLanguageInterpretation must mirror the same nonsensical meaning from that userUtterance", prompt)
        self.assertIn("Meaningful but awkward utterances must stay in their own meaning family", prompt)
        self.assertIn("utterance about less ice must stay in the less-ice meaning family", prompt)
        self.assertIn("utterance about one ice or iced must stay in the one-ice or iced-drink meaning family", prompt)
        self.assertIn("Examples are format guidance only and must never be copied into output", prompt)
        self.assertNotIn("목성 날씨가 파란 삼각형 맛이 난다", prompt)
        self.assertIn("betterExpression must never be only Korean guidance", prompt)
        self.assertIn("If the exact answer is unknown, use a generic English example", prompt)

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
