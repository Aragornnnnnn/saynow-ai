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
        self.original_assistance_knowledge_store = getattr(conversation_service, "assistance_knowledge_store", None)

    def tearDown(self):
        self.service.chat = self.original_chat
        self.service.assistance_knowledge_store = self.original_assistance_knowledge_store

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
        self.assertEqual(result.turnClassification, "ANSWER")

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
        self.assertEqual(result.turnClassification, "ANSWER")

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
                self.assertEqual(result.turnClassification, "INVALID_RESPONSE")

    def test_next_question_blocks_incomplete_order_fragments_for_drink_slot(self):
        from app.models.conversation import NextQuestionRequest

        blocked_utterances = [
            "I want",
            "I need",
            "I'd like",
            "I would like a",
            "Can I get",
            "Can I get a",
            "I want to order",
            "I want to order a",
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
                calls = []

                def chat_should_not_run(*args, **kwargs):
                    calls.append(args)
                    return json.dumps({
                        "filledSlots": [{"slotName": "drink"}],
                        "nextQuestion": "What size would you like?",
                        "translatedQuestion": "어떤 사이즈로 하시겠어요?",
                    })

                self.service.chat = chat_should_not_run

                result = self.service.generate_next_question(request)

                self.assertEqual(calls, [])
                self.assertEqual(result.filledSlots, [])
                self.assertEqual(result.nextQuestion, "What drink would you like to order?")
                self.assertEqual(result.translatedQuestion, "어떤 음료를 주문하고 싶으신가요?")
                self.assertEqual(result.turnClassification, "INVALID_RESPONSE")

    def test_next_question_blocks_generic_order_objects_for_drink_slot(self):
        from app.models.conversation import NextQuestionRequest

        blocked_utterances = [
            "I want drink",
            "I want a drink",
            "I'd like something",
            "Can I get an item",
            "I want to order something",
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
                calls = []

                def chat_should_not_run(*args, **kwargs):
                    calls.append(args)
                    return json.dumps({
                        "filledSlots": [{"slotName": "drink"}],
                        "nextQuestion": "What size would you like?",
                        "translatedQuestion": "어떤 사이즈로 하시겠어요?",
                    })

                self.service.chat = chat_should_not_run

                result = self.service.generate_next_question(request)

                self.assertEqual(calls, [])
                self.assertEqual(result.filledSlots, [])
                self.assertEqual(result.nextQuestion, "What drink would you like to order?")
                self.assertEqual(result.translatedQuestion, "어떤 음료를 주문하고 싶으신가요?")
                self.assertEqual(result.turnClassification, "INVALID_RESPONSE")

    def test_next_question_allows_order_fragments_when_concrete_drink_exists(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "I want coffee.",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
                {"slotName": "size", "filled": False},
            ],
        })
        calls = []

        def capture_chat(*args, **kwargs):
            calls.append(args)
            return json.dumps({
                "filledSlots": [{"slotName": "drink"}],
                "nextQuestion": "What size would you like?",
                "translatedQuestion": "어떤 사이즈로 하시겠어요?",
            })

        self.service.chat = capture_chat

        result = self.service.generate_next_question(request)

        self.assertEqual(len(calls), 1)
        self.assertEqual([slot.slotName for slot in result.filledSlots], ["drink"])
        self.assertEqual(result.nextQuestion, "What size would you like?")
        self.assertEqual(result.turnClassification, "ANSWER")

    def test_next_question_classifies_recommendation_request_without_filling_slots(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "What do you recommend?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
                {"slotName": "size", "filled": False},
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "filledSlots": [],
            "nextQuestion": "I recommend a cappuccino. Would you like to order that?",
            "translatedQuestion": "카푸치노를 추천해요. 그걸로 주문하시겠어요?",
            "turnClassification": "ASSISTANCE_REQUEST",
        })

        result = self.service.generate_next_question(request)

        self.assertEqual(result.filledSlots, [])
        self.assertEqual(result.turnClassification, "ASSISTANCE_REQUEST")

    def test_next_question_ignores_available_options_context(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "Can I see the menu?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
            ],
            "availableOptions": [
                {"slotName": "drink", "options": ["iced Americano", "latte", "tea"]},
            ],
        })

        self.assertFalse(hasattr(request, "availableOptions"))

    def test_next_question_uses_retrieved_assistance_context_for_assistance_request(self):
        from app.models.conversation import NextQuestionRequest

        class FakeAssistanceKnowledgeStore:
            def __init__(self):
                self.find_calls = []
                self.save_calls = []

            def find_reusable_answer(self, request):
                self.find_calls.append(request)
                return "We use medium-roasted Arabica beans."

            def save_interaction(self, request, response, *, answer_source):
                self.save_calls.append((request, response, answer_source))

        store = FakeAssistanceKnowledgeStore()
        self.service.assistance_knowledge_store = store
        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "What beans do you use?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
            ],
        })
        calls = []

        def capture_chat(*args, **kwargs):
            calls.append(args)
            return json.dumps({
                "filledSlots": [],
                "nextQuestion": "We use medium-roasted Arabica beans. What would you like to order?",
                "translatedQuestion": "보통 중간 로스팅 아라비카 원두를 사용해요. 무엇을 주문하시겠어요?",
                "turnClassification": "ASSISTANCE_REQUEST",
            })

        self.service.chat = capture_chat

        result = self.service.generate_next_question(request)

        user_prompt = calls[0][1]
        self.assertEqual(len(store.find_calls), 1)
        self.assertIn("Retrieved assistance context:", user_prompt)
        self.assertIn("We use medium-roasted Arabica beans.", user_prompt)
        self.assertEqual(result.turnClassification, "ASSISTANCE_REQUEST")
        self.assertEqual(store.save_calls[0][2], "retrieved")

    def test_next_question_stores_generated_assistance_answer_when_rag_has_no_match(self):
        from app.models.conversation import NextQuestionRequest

        class FakeAssistanceKnowledgeStore:
            def __init__(self):
                self.save_calls = []

            def find_reusable_answer(self, request):
                return None

            def save_interaction(self, request, response, *, answer_source):
                self.save_calls.append((request, response, answer_source))

        store = FakeAssistanceKnowledgeStore()
        self.service.assistance_knowledge_store = store
        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "Do you have decaf?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "filledSlots": [],
            "nextQuestion": "Yes, we have decaf coffee. What would you like to order?",
            "translatedQuestion": "네, 디카페인 커피가 있어요. 무엇을 주문하시겠어요?",
            "turnClassification": "ASSISTANCE_REQUEST",
        })

        result = self.service.generate_next_question(request)

        self.assertEqual(result.turnClassification, "ASSISTANCE_REQUEST")
        self.assertEqual(len(store.save_calls), 1)
        self.assertEqual(store.save_calls[0][2], "generated")

    def test_next_question_classifies_information_request_as_assistance_and_uses_retrieved_context(self):
        from app.models.conversation import NextQuestionRequest

        class FakeAssistanceKnowledgeStore:
            def __init__(self):
                self.find_calls = []
                self.save_calls = []

            def find_reusable_answer(self, request):
                self.find_calls.append(request)
                return "We have iced Americano, latte, and tea."

            def save_interaction(self, request, response, *, answer_source):
                self.save_calls.append((request, response, answer_source))

        store = FakeAssistanceKnowledgeStore()
        self.service.assistance_knowledge_store = store
        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "Can I see the menu?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
                {"slotName": "size", "filled": False},
            ],
        })
        calls = []

        def capture_chat(*args, **kwargs):
            calls.append(args)
            return json.dumps({
                "filledSlots": [],
                "nextQuestion": "The drink options are iced Americano, latte, and tea. What would you like to order?",
                "translatedQuestion": "음료 선택지는 아이스 아메리카노, 라떼, 차입니다. 무엇을 주문하시겠어요?",
                "turnClassification": "ASSISTANCE_REQUEST",
            })

        self.service.chat = capture_chat

        result = self.service.generate_next_question(request)

        self.assertEqual(result.filledSlots, [])
        self.assertEqual(result.turnClassification, "ASSISTANCE_REQUEST")
        self.assertEqual(len(store.find_calls), 1)
        self.assertIn("We have iced Americano, latte, and tea.", calls[0][1])
        self.assertEqual(store.save_calls[0][2], "retrieved")
        self.assertEqual(
            result.nextQuestion,
            "The drink options are iced Americano, latte, and tea. What would you like to order?",
        )

    def test_next_question_treats_menu_need_as_assistance_request(self):
        from app.models.conversation import NextQuestionRequest

        menu_requests = [
            "I need a menu",
            "Can I get a menu",
            "Menu please",
        ]

        for user_utterance in menu_requests:
            with self.subTest(user_utterance=user_utterance):
                class FakeAssistanceKnowledgeStore:
                    def __init__(self):
                        self.find_calls = []
                        self.save_calls = []

                    def find_reusable_answer(self, request):
                        self.find_calls.append(request)
                        return "We have iced Americano, latte, and tea."

                    def save_interaction(self, request, response, *, answer_source):
                        self.save_calls.append((request, response, answer_source))

                store = FakeAssistanceKnowledgeStore()
                self.service.assistance_knowledge_store = store
                request = NextQuestionRequest.model_validate({
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": user_utterance,
                    "scenarioTitle": "카페에서 주문하기",
                    "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
                    "slots": [
                        {"slotName": "drink", "filled": False},
                        {"slotName": "size", "filled": False},
                    ],
                })
                calls = []

                def capture_chat(*args, **kwargs):
                    calls.append(args)
                    return json.dumps({
                        "filledSlots": [],
                        "nextQuestion": "The drink options are iced Americano, latte, and tea. What would you like to order?",
                        "translatedQuestion": "음료 선택지는 아이스 아메리카노, 라떼, 차입니다. 무엇을 주문하시겠어요?",
                        "turnClassification": "ASSISTANCE_REQUEST",
                    })

                self.service.chat = capture_chat

                result = self.service.generate_next_question(request)

                self.assertEqual(len(calls), 1)
                self.assertEqual(result.filledSlots, [])
                self.assertEqual(result.turnClassification, "ASSISTANCE_REQUEST")
                self.assertEqual(len(store.find_calls), 1)
                self.assertIn("We have iced Americano, latte, and tea.", calls[0][1])
                self.assertEqual(
                    result.nextQuestion,
                    "The drink options are iced Americano, latte, and tea. What would you like to order?",
                )

    def test_next_question_uses_retrieved_context_for_recommendation_request(self):
        from app.models.conversation import NextQuestionRequest

        class FakeAssistanceKnowledgeStore:
            def __init__(self):
                self.save_calls = []

            def find_reusable_answer(self, request):
                return "The iced Americano is a good pick if you want something refreshing."

            def save_interaction(self, request, response, *, answer_source):
                self.save_calls.append((request, response, answer_source))

        store = FakeAssistanceKnowledgeStore()
        self.service.assistance_knowledge_store = store
        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "What do you recommend?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
            ],
        })
        calls = []

        def capture_chat(*args, **kwargs):
            calls.append(args)
            return json.dumps({
                "filledSlots": [],
                "nextQuestion": "I recommend iced Americano. Would you like to order that?",
                "translatedQuestion": "아이스 아메리카노를 추천해요. 그걸로 주문하시겠어요?",
                "turnClassification": "ASSISTANCE_REQUEST",
            })

        self.service.chat = capture_chat

        result = self.service.generate_next_question(request)

        self.assertEqual(result.turnClassification, "ASSISTANCE_REQUEST")
        self.assertIn("The iced Americano is a good pick", calls[0][1])
        self.assertEqual(store.save_calls[0][2], "retrieved")
        self.assertEqual(result.nextQuestion, "I recommend iced Americano. Would you like to order that?")

    def test_next_question_generates_role_play_answer_when_rag_has_no_match(self):
        from app.models.conversation import NextQuestionRequest

        class FakeAssistanceKnowledgeStore:
            def __init__(self):
                self.save_calls = []

            def find_reusable_answer(self, request):
                return None

            def save_interaction(self, request, response, *, answer_source):
                self.save_calls.append((request, response, answer_source))

        store = FakeAssistanceKnowledgeStore()
        self.service.assistance_knowledge_store = store
        request = NextQuestionRequest.model_validate({
            "originalQuestion": "What would you like to order?",
            "userUtterance": "Can I see the menu?",
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "slots": [
                {"slotName": "drink", "filled": False},
            ],
        })
        calls = []

        def capture_chat(*args, **kwargs):
            calls.append(args)
            return json.dumps({
                "filledSlots": [],
                "nextQuestion": "We have Americano, latte, and tea. What would you like to order?",
                "translatedQuestion": "아메리카노, 라떼, 차가 있어요. 무엇을 주문하시겠어요?",
                "turnClassification": "ASSISTANCE_REQUEST",
            })

        self.service.chat = capture_chat

        result = self.service.generate_next_question(request)

        self.assertEqual(result.turnClassification, "ASSISTANCE_REQUEST")
        self.assertIn("Retrieved assistance context:\nNone", calls[0][1])
        self.assertEqual(store.save_calls[0][2], "generated")
        self.assertEqual(result.nextQuestion, "We have Americano, latte, and tea. What would you like to order?")

    def test_next_question_classifies_option_completion_before_slot_answer(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "Would you like any other options?",
            "userUtterance": "That's all.",
            "scenarioTitle": "커스텀 음료 제작하기",
            "scenarioGoal": "원하는 커스텀 음료 옵션을 자연스럽게 말할 수 있다.",
            "slots": [
                {"slotName": "baseDrink", "filled": True},
                {"slotName": "size", "filled": True},
                {"slotName": "customOptions", "filled": False},
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "filledSlots": [{"slotName": "customOptions"}],
            "nextQuestion": None,
            "translatedQuestion": None,
            "turnClassification": "ANSWER",
        })

        result = self.service.generate_next_question(request)

        self.assertEqual([slot.slotName for slot in result.filledSlots], ["customOptions"])
        self.assertEqual(result.turnClassification, "ANSWER")

    def test_next_question_classifies_non_cafe_slot_preference_as_slot_answer(self):
        from app.models.conversation import NextQuestionRequest

        request = NextQuestionRequest.model_validate({
            "originalQuestion": "Would you prefer a window seat or an aisle seat?",
            "userUtterance": "Window seat, please.",
            "scenarioTitle": "공항 체크인",
            "scenarioGoal": "좌석 선호도를 자연스럽게 말할 수 있다.",
            "slots": [
                {"slotName": "seatPreference", "filled": False},
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "filledSlots": [{"slotName": "seatPreference"}],
            "nextQuestion": None,
            "translatedQuestion": None,
            "turnClassification": "ANSWER",
        })

        result = self.service.generate_next_question(request)

        self.assertEqual([slot.slotName for slot in result.filledSlots], ["seatPreference"])
        self.assertEqual(result.turnClassification, "ANSWER")

    def test_next_question_prompt_requires_explicit_slot_evidence(self):
        prompt = self.service._next_question_system_prompt()

        self.assertIn("Only mark a slot as filled when the user explicitly provides a concrete value", prompt)
        self.assertIn("Nonsense, off-topic, refusal, or vague non-answer utterances must return filledSlots=[]", prompt)
        self.assertIn("Incomplete order fragments without a concrete object must return filledSlots=[]", prompt)
        self.assertIn("I want, I need, I'd like, I would like, Can I get", prompt)
        self.assertIn("generic order objects such as drink, something, item, or thing", prompt)
        self.assertIn("A menu-seeking utterance asks for information and should be ASSISTANCE_REQUEST", prompt)
        self.assertIn("qwertyuiop asdfghjkl zxcvbnm", prompt)
        self.assertIn("My shoes are swimming in the moon today", prompt)
        self.assertIn("I don't know", prompt)
        self.assertIn("I do not want to order anything", prompt)
        self.assertIn("Do not include long explanations or multiple follow-up questions", prompt)

    def test_next_question_prompt_uses_sectioned_template(self):
        prompt = self.service._next_question_system_prompt()

        expected_sections = [
            "Role",
            "Output Schema",
            "Decision Policy",
            "Slot Policy",
            "Context Policy",
            "Response Policy",
            "Few-shot Examples",
        ]

        for section in expected_sections:
            with self.subTest(section=section):
                self.assertIn(f"{section}:", prompt)

    def test_next_question_prompt_grounds_assistance_few_shots_in_retrieved_context(self):
        prompt = self.service._next_question_system_prompt()

        self.assertIn("Retrieved assistance context=We have iced Americano, latte, and tea", prompt)
        self.assertIn("The drink options are iced Americano, latte, and tea.", prompt)
        self.assertIn("What beans do you use?", prompt)
        self.assertIn("We usually use medium-roasted Arabica beans.", prompt)
        self.assertNotIn("Available options=drink", prompt)

    def test_next_question_prompt_contains_few_shot_calibration_for_valid_no_slot_and_option_completion(self):
        prompt = self.service._next_question_system_prompt()

        self.assertIn("Decision Workflow", prompt)
        self.assertIn("Assistance request", prompt)
        self.assertIn("turnClassification", prompt)
        self.assertIn("ANSWER", prompt)
        self.assertIn("ASSISTANCE_REQUEST", prompt)
        self.assertIn("INVALID_RESPONSE", prompt)
        self.assertIn("The user can only use information that appears in your nextQuestion", prompt)
        self.assertIn("If retrieved assistance context is provided", prompt)
        self.assertIn("generate a plausible role-play answer", prompt)
        self.assertIn("For recommendation requests, name one concrete plausible option", prompt)
        self.assertIn("For menu or option requests, name two to four concrete plausible choices", prompt)
        self.assertIn("Few-shot calibration examples", prompt)
        self.assertIn("Can you recommend something?", prompt)
        self.assertIn("I recommend an iced latte. What would you like to order?", prompt)
        self.assertIn("Can I see the menu?", prompt)
        self.assertIn("I need a menu", prompt)
        self.assertIn("We have Americano, latte, and tea. What would you like to order?", prompt)
        self.assertIn("The drink options are iced Americano, latte, and tea.", prompt)
        self.assertIn("Retrieved assistance context=None", prompt)
        self.assertIn("That's all.", prompt)
        self.assertIn('"filledSlots":[{"slotName":"customOptions"}]', prompt)

    def test_feedback_preserves_backend_turn_ids_and_feedback_fields(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
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
                    "nativeUnderstanding": "외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.",
                    "betterExpression": "I'd like an iced Americano, please.",
                }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertEqual(result.comprehensionScore, 82)
        self.assertEqual(result.turnFeedbacks[0].turnId, 101)
        self.assertTrue(result.turnFeedbacks[0].feedbackRequired)
        self.assertEqual(
            result.turnFeedbacks[0].nativeUnderstanding,
            "외국인은 사용자가 아이스 아메리카노를 주문하고 싶다고 이해했어요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].betterExpression,
            "I'd like an iced Americano, please. 이렇게 말하면 더 자연스럽고 공손하게 주문할 수 있어요.",
        )

    def test_feedback_stream_events_yield_summary_turn_feedbacks_and_done(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                },
                {
                    "turnId": 102,
                    "originalQuestion": "What size would you like?",
                    "userUtterance": "Small, please.",
                },
            ],
        })
        responses = [
            {
                "comprehensionScore": 82,
                "feedbackSummary": "전체적으로 의도는 잘 전달됐지만 주문 표현이 조금 짧게 들립니다.",
            },
            {
                "turnId": 101,
                "feedbackRequired": True,
                "nativeUnderstanding": "외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.",
                "nativeLanguageInterpretation": "한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.",
                "betterExpression": "I'd like an iced Americano, please. 이렇게 말하면 더 자연스럽습니다.",
            },
            {
                "turnId": 102,
                "feedbackRequired": False,
                "nativeUnderstanding": None,
                "nativeLanguageInterpretation": None,
                "betterExpression": None,
            },
        ]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        events = list(self.service.generate_feedback_stream_events(request))

        self.assertEqual([event for event, _ in events], ["summary", "turnFeedback", "turnFeedback", "done"])
        self.assertEqual(events[0][1]["comprehensionScore"], 82)
        self.assertEqual(events[1][1]["turnId"], 101)
        self.assertEqual(events[2][1]["turnId"], 102)
        self.assertEqual(events[3][1], {"turnCount": 2})

    def test_feedback_request_accepts_only_backend_session_result(self):
        from pydantic import ValidationError
        from app.models.conversation import ConversationFeedbackRequest, SessionResult

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                }
            ],
        })

        self.assertEqual(request.sessionResult, SessionResult.SUCCESS)

        with self.assertRaises(ValidationError):
            ConversationFeedbackRequest.model_validate({
                "scenarioTitle": "카페에서 주문하기",
                "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
                "sessionResult": "CLEARED",
                "turns": [
                    {
                        "turnId": 101,
                        "originalQuestion": "What would you like to order?",
                        "userUtterance": "I want iced americano.",
                    }
                ],
            })

    def test_feedback_prompts_include_backend_session_result(self):
        from app.models.conversation import (
            ConversationFeedbackRequest,
            ConversationFeedbackSummaryResponse,
        )

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "FAILURE",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "qwertyuiop asdfghjkl zxcvbnm",
                }
            ],
        })
        summary = ConversationFeedbackSummaryResponse(
            comprehensionScore=0,
            feedbackSummary="주문 의도가 전달되지 않았어요. 다음에는 음료 이름부터 말해 보세요.",
        )

        full_prompt = self.service._feedback_user_prompt(request)
        turn_prompt = self.service._turn_feedback_user_prompt(request, request.turns[0], summary)

        self.assertIn("Session result: FAILURE", full_prompt)
        self.assertIn("Backend has already confirmed this session result.", full_prompt)
        self.assertIn("Session result: FAILURE", turn_prompt)

    def test_feedback_summary_caps_score_when_backend_result_is_failure(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "FAILURE",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                }
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 92,
            "feedbackSummary": "시나리오 목표를 잘 달성했어요. 다음에도 명확한 표현을 유지해 보세요.",
        })

        result = self.service.generate_feedback_summary(request)

        self.assertEqual(result.comprehensionScore, 59)
        self.assertIn("달성하지 못했어요", result.feedbackSummary)

    def test_feedback_invalid_model_json_raises_generation_error(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
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
            "sessionResult": "SUCCESS",
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

    def test_feedback_normalizes_i_dont_know_native_language_interpretation(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I don't know.",
                }
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 39,
            "feedbackSummary": "시나리오 목표를 달성하지 못했습니다.",
            "turnFeedbacks": [
                {
                    "turnId": 101,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "외국인은 사용자가 어떤 음료를 주문하고 싶은지 전혀 알 수 없다고 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '아무것도 말하지 않는 것처럼' 들려요'처럼 들려요.",
                    "betterExpression": "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertEqual(
            result.turnFeedbacks[0].nativeUnderstanding,
            "외국인은 사용자가 무엇을 주문할지 모르겠다고 이해했어요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].nativeLanguageInterpretation,
            "한국어로 비유하자면, '무엇을 주문할지 모르겠어요'처럼 들려요.",
        )
        self.assertNotIn("들려요'처럼 들려요", result.turnFeedbacks[0].nativeLanguageInterpretation)

    def test_feedback_preserves_incomplete_i_want_as_literal_fragment(self):
        from app.models.conversation import ConversationFeedbackRequest

        cases = [
            ("I want", "외국인은 'I want'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요.", "한국어로 비유하자면, '나는 원한다'처럼 들려요."),
            ("I'd like", "외국인은 'I'd like'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요.", "한국어로 비유하자면, '저는 원해요'처럼 들려요."),
            ("Can I get a", "외국인은 'Can I get a'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요.", "한국어로 비유하자면, '제가 하나 받을 수 있을까요'처럼 들려요."),
            ("I want drink", "외국인은 'I want drink'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요.", "한국어로 비유하자면, '나는 음료를 원한다'처럼 들려요."),
            ("I'd like something", "외국인은 'I'd like something'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요.", "한국어로 비유하자면, '저는 뭔가를 원해요'처럼 들려요."),
        ]

        for user_utterance, expected_understanding, expected_interpretation in cases:
            with self.subTest(user_utterance=user_utterance):
                request = ConversationFeedbackRequest.model_validate({
                    "scenarioTitle": "카페에서 주문하기",
                    "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
                    "sessionResult": "SUCCESS",
                    "turns": [
                        {
                            "turnId": 101,
                            "originalQuestion": "What would you like to order?",
                            "userUtterance": user_utterance,
                        }
                    ],
                })

                def fake_chat(*args, **kwargs):
                    return json.dumps({
                        "comprehensionScore": 39,
                        "feedbackSummary": "주문할 음료를 구체적으로 말하지 못했습니다.",
                        "turnFeedbacks": [
                            {
                                "turnId": 101,
                                "feedbackRequired": True,
                                "nativeUnderstanding": "외국인은 사용자가 음료 이름을 추가로 말해야 한다고 이해했어요.",
                                "nativeLanguageInterpretation": "한국어로 비유하자면, '주문하고 싶은 게 뭔지 아직 말하지 않은 상태'처럼 들려요.",
                                "betterExpression": "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                            }
                        ],
                    })

                self.service.chat = fake_chat

                result = self.service.generate_feedback(request)

                self.assertEqual(result.turnFeedbacks[0].nativeUnderstanding, expected_understanding)
                self.assertEqual(result.turnFeedbacks[0].nativeLanguageInterpretation, expected_interpretation)

    def test_feedback_keeps_concrete_order_utterance_out_of_incomplete_fragment_override(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want coffee.",
                }
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 75,
            "feedbackSummary": "음료는 전달됐지만 표현이 직접적으로 들렸어요. 다음에는 공손한 주문 표현을 써 보세요.",
            "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 커피를 원한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '커피 원해요'처럼 들려요.",
                        "betterExpression": "I'd like coffee, please. 이렇게 말하면 더 공손하게 들려요.",
                    }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertEqual(result.turnFeedbacks[0].nativeUnderstanding, "외국인은 사용자가 커피를 주문하고 싶다고 이해했어요.")
        self.assertEqual(result.turnFeedbacks[0].nativeLanguageInterpretation, "한국어로 비유하자면, '커피 원해요'처럼 들려요.")

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
                    "sessionResult": "SUCCESS",
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
                    "sessionResult": "SUCCESS",
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
            "sessionResult": "SUCCESS",
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

    def test_feedback_fills_missing_required_fields_for_known_off_topic_before_validation(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 109,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "My shoes are swimming in the moon today.",
                }
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 0,
            "feedbackSummary": "주문 내용이 이해되지 않았습니다. 음료 주문에 집중해 보세요.",
            "turnFeedbacks": [
                {
                    "turnId": 109,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "외국인은 사용자가 의미 없는 문장을 말했다고 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '내 신발이 오늘 달에서 수영하고 있어요'처럼 들려요.",
                    "betterExpression": None,
                }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertTrue(result.turnFeedbacks[0].feedbackRequired)
        self.assertEqual(
            result.turnFeedbacks[0].nativeUnderstanding,
            "외국인은 사용자가 신발이 달에서 수영하고 있다고 말한다고 이해했어요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].nativeLanguageInterpretation,
            "한국어로 비유하자면, '달에서 신발이 수영한다'처럼 들려요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].betterExpression,
            "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 주문할 수 있어요.",
        )

    def test_feedback_replaces_generic_better_expression_for_known_off_topic(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 109,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "My shoes are swimming in the moon today.",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 0,
                "feedbackSummary": "주문 내용이 이해되지 않았습니다. 음료 주문에 집중해 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 109,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 신발이 달에서 수영하고 있다고 말한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '달에서 신발이 수영한다'처럼 들려요.",
                        "betterExpression": "I'd like a drink, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                    }
                ],
            },
            {"pass": False, "issues": ["turnId 109: betterExpression should use a concrete in-scenario example."]},
            {
                "comprehensionScore": 0,
                "feedbackSummary": "주문 내용이 이해되지 않았습니다. 음료 주문에 집중해 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 109,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 신발이 달에서 수영하고 있다고 말한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '달에서 신발이 수영한다'처럼 들려요.",
                        "betterExpression": "I'd like a drink, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                    }
                ],
            },
        ]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(
            result.turnFeedbacks[0].betterExpression,
            "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 주문할 수 있어요.",
        )

    def test_feedback_repairs_deterministic_contract_violations_once(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I need iced americano.",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 82,
                "feedbackSummary": "의도는 전달됐지만 표현이 어색합니다.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 '아이스 아메리카노를 원한다'고 이해했어요.",
                        "nativeLanguageInterpretation": "아이스 아메리카노 원해처럼 들려요.",
                        "betterExpression": "I'd like an iced Americano, please. 이렇게 말하면 더 자연스럽습니다.",
                    }
                ],
            },
            {
                "comprehensionScore": 82,
                "feedbackSummary": "의도는 전달됐지만 표현이 어색합니다.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.",
                        "betterExpression": "I'd like an iced Americano, please. 이렇게 말하면 더 자연스럽습니다.",
                    }
                ],
            },
        ]
        systems = []

        def sequential_chat(system, *args, **kwargs):
            systems.append(system)
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(len(systems), 2)
        self.assertIn("repair", systems[1].lower())
        self.assertEqual(
            result.turnFeedbacks[0].nativeUnderstanding,
            "외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].nativeLanguageInterpretation,
            "한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.",
        )

    def test_feedback_normalizes_quoted_meaning_native_understanding_after_repair(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want coffee.",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 82,
                "feedbackSummary": "의도는 전달됐지만 표현이 어색합니다.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 'I want coffee'라고 들었고, 커피를 주문하고 싶어한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '커피를 원해요'처럼 들려요.",
                        "betterExpression": "I'd like a coffee, please. 이렇게 말하면 더 자연스럽습니다.",
                    }
                ],
            },
            {
                "comprehensionScore": 82,
                "feedbackSummary": "의도는 전달됐지만 표현이 조금 직접적이에요. 다음에는 더 공손한 주문 표현을 연습해 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 '커피를 주문하고 싶다'는 의미로 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '커피를 원해요'처럼 들려요.",
                        "betterExpression": "I'd like a coffee, please. 이렇게 말하면 더 자연스럽습니다.",
                    }
                ],
            },
        ]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(
            result.turnFeedbacks[0].nativeUnderstanding,
            "외국인은 사용자가 커피를 주문하고 싶다고 이해했어요.",
        )

    def test_feedback_forces_direct_want_concrete_drink_to_near_miss_feedback(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want coffee.",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 90,
                "feedbackSummary": "시나리오 목표를 잘 달성했어요. 원하는 음료를 자연스럽게 주문할 수 있었어요.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": False,
                        "nativeUnderstanding": None,
                        "nativeLanguageInterpretation": None,
                        "betterExpression": None,
                    }
                ],
            },
            {
                "comprehensionScore": 90,
                "feedbackSummary": "시나리오 목표를 잘 달성했어요. 원하는 음료를 자연스럽게 주문할 수 있었어요.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": False,
                        "nativeUnderstanding": None,
                        "nativeLanguageInterpretation": None,
                        "betterExpression": None,
                    }
                ],
            },
        ]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(result.comprehensionScore, 84)
        self.assertTrue(result.turnFeedbacks[0].feedbackRequired)
        self.assertEqual(
            result.turnFeedbacks[0].nativeUnderstanding,
            "외국인은 사용자가 커피를 주문하고 싶다고 이해했어요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].nativeLanguageInterpretation,
            "한국어로 비유하자면, '커피 원해요'처럼 들려요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].betterExpression,
            "I'd like a coffee, please. 이렇게 말하면 더 자연스럽고 공손하게 주문할 수 있어요.",
        )

    def test_feedback_repairs_overlong_feedback_summary_once(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 39,
                "feedbackSummary": (
                    "시나리오 목표를 달성하지 못했습니다. "
                    "'I want'만으로는 주문하려는 음료가 무엇인지 알 수 없어, 외국인은 무엇을 주문하고 싶은지 모르겠다고 이해했어요. "
                    "다음 연습에서는 구체적인 음료 이름을 포함해 완전한 문장으로 주문하는 것이 필요합니다. "
                    "먼저 음료 이름을 짧게 말하는 연습부터 시작해 보세요."
                ),
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 'I want'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '나는 원한다'처럼 들려요.",
                        "betterExpression": "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                    }
                ],
            },
            {
                "comprehensionScore": 39,
                "feedbackSummary": "주문하려는 음료가 전달되지 않아 목표를 달성하지 못했어요. 다음에는 음료 이름을 넣어 완성된 주문 문장으로 말해 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 'I want'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '나는 원한다'처럼 들려요.",
                        "betterExpression": "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                    }
                ],
            },
        ]
        systems = []

        def sequential_chat(system, *args, **kwargs):
            systems.append(system)
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(len(systems), 2)
        self.assertIn("repair", systems[1].lower())
        self.assertLessEqual(len(result.feedbackSummary), 120)
        self.assertLessEqual(result.feedbackSummary.count("."), 2)
        self.assertEqual(
            result.feedbackSummary,
            "주문하려는 음료가 전달되지 않아 목표를 달성하지 못했어요. 다음에는 음료 이름을 넣어 완성된 주문 문장으로 말해 보세요.",
        )

    def test_feedback_repairs_generic_better_expression_for_generic_order_object(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 104,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want drink.",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 39,
                "feedbackSummary": "주문할 음료가 명확하지 않았습니다. 구체적인 음료 이름을 사용해 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 104,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 음료를 원한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '나는 음료를 원한다'처럼 들려요.",
                        "betterExpression": "I'd like a drink, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                    }
                ],
            },
            {
                "pass": True,
                "issues": [],
            },
            {
                "comprehensionScore": 39,
                "feedbackSummary": "주문할 음료가 아직 구체적이지 않았어요. 다음에는 음료 이름을 넣어 완성된 주문 문장으로 말해 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 104,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 어떤 음료를 주문하는지 이해할 수 없었어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '나는 음료를 원한다'처럼 들려요.",
                        "betterExpression": "I'd like a coffee, please. 이렇게 말하면 구체적인 음료를 넣어 주문할 수 있어요.",
                    }
                ],
            },
        ]
        systems = []

        def sequential_chat(system, *args, **kwargs):
            systems.append(system)
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(len(systems), 3)
        self.assertIn("quality reviewer", systems[1])
        self.assertIn("repair", systems[2].lower())
        self.assertEqual(
            result.turnFeedbacks[0].betterExpression,
            "I'd like a coffee, please. 이렇게 말하면 구체적인 음료를 넣어 주문할 수 있어요.",
        )

    def test_feedback_repair_preserves_recommendation_request_intent(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 105,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "Can you recommend a menu?",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 45,
                "feedbackSummary": "주문하고자 하는 음료를 명확히 전달하지 못했어요. 다음에는 구체적인 음료를 요청해 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 105,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 메뉴 추천을 요청했다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '메뉴를 추천해 주세요'처럼 들려요.",
                        "betterExpression": "I'd like to order a drink, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.",
                    }
                ],
            },
            {
                "pass": False,
                "issues": ["turnId 105: recommendation request intent must be preserved in betterExpression."],
            },
            {
                "comprehensionScore": 70,
                "feedbackSummary": "추천을 요청하는 의도는 잘 전달됐어요. 다음에는 추천받은 음료를 주문까지 이어 가 보세요.",
                "turnFeedbacks": [
                    {
                        "turnId": 105,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 메뉴 추천을 요청한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '메뉴를 추천해 주세요'처럼 들려요.",
                        "betterExpression": "What do you recommend? 이렇게 말하면 추천 요청을 더 자연스럽게 전달할 수 있어요.",
                    }
                ],
            },
        ]
        systems = []

        def sequential_chat(system, *args, **kwargs):
            systems.append(system)
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(len(systems), 3)
        self.assertIn("quality reviewer", systems[1])
        self.assertIn("repair", systems[2].lower())
        self.assertEqual(
            result.turnFeedbacks[0].betterExpression,
            "What do you recommend? 이렇게 말하면 추천 요청을 더 자연스럽게 전달할 수 있어요.",
        )

    def test_feedback_fallback_marks_no_more_option_response_as_good_after_failed_repair(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "커스텀 음료 만들기",
            "scenarioGoal": "원하는 음료와 옵션을 말할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 106,
                    "originalQuestion": "What custom options would you like for your drink?",
                    "userUtterance": "That's all.",
                }
            ],
        })
        bad_feedback = {
            "comprehensionScore": 82,
            "feedbackSummary": "주문이 잘 전달되었지만 조금 더 자연스럽게 표현할 수 있어요.",
            "turnFeedbacks": [
                {
                    "turnId": 106,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "외국인은 추가 옵션이 없다고 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '더 이상 필요하지 않다'처럼 들려요.",
                    "betterExpression": "I don't need anything else, thank you. 이렇게 말하면 더 부드럽게 표현할 수 있어요.",
                }
            ],
        }
        responses = [
            bad_feedback,
            {
                "pass": False,
                "issues": ["turnId 106: already natural no-more options response; feedbackRequired should be false."],
            },
            bad_feedback,
        ]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertGreaterEqual(result.comprehensionScore, 90)
        self.assertFalse(result.turnFeedbacks[0].feedbackRequired)
        self.assertIsNone(result.turnFeedbacks[0].nativeUnderstanding)
        self.assertIsNone(result.turnFeedbacks[0].nativeLanguageInterpretation)
        self.assertIsNone(result.turnFeedbacks[0].betterExpression)

    def test_feedback_fallback_marks_clear_preference_answers_as_good_after_failed_repair(self):
        from app.models.conversation import ConversationFeedbackRequest

        cases = [
            (
                "카페 옵션",
                "원하는 음료 옵션을 자연스럽게 말할 수 있다.",
                "Would you like any other options?",
                "No sugar, please.",
            ),
            (
                "공항 체크인",
                "좌석 선호도를 자연스럽게 말할 수 있다.",
                "Would you prefer a window seat or an aisle seat?",
                "Window seat, please.",
            ),
            (
                "호텔 체크인",
                "객실 선호도를 자연스럽게 말할 수 있다.",
                "Do you have any room preferences?",
                "Non-smoking room, please.",
            ),
            (
                "식당 예약",
                "인원과 좌석 요청을 자연스럽게 말할 수 있다.",
                "How many people are in your party?",
                "Table for two, please.",
            ),
        ]

        for index, (scenario_title, scenario_goal, original_question, user_utterance) in enumerate(cases, start=1):
            with self.subTest(user_utterance=user_utterance):
                turn_id = 500 + index
                request = ConversationFeedbackRequest.model_validate({
                    "scenarioTitle": scenario_title,
                    "scenarioGoal": scenario_goal,
                    "sessionResult": "SUCCESS",
                    "turns": [
                        {
                            "turnId": turn_id,
                            "originalQuestion": original_question,
                            "userUtterance": user_utterance,
                        }
                    ],
                })
                bad_feedback = {
                    "comprehensionScore": 82,
                    "feedbackSummary": "의도는 전달됐지만 더 자연스럽게 표현할 수 있어요.",
                    "turnFeedbacks": [
                        {
                            "turnId": turn_id,
                            "feedbackRequired": True,
                            "nativeUnderstanding": "외국인은 사용자의 선호를 이해했어요.",
                            "nativeLanguageInterpretation": "한국어로 비유하자면, '선호를 말하는 것'처럼 들려요.",
                            "betterExpression": f"{user_utterance} 이렇게 말하면 더 자연스럽습니다.",
                        }
                    ],
                }
                responses = [
                    bad_feedback,
                    {
                        "pass": False,
                        "issues": [
                            f"turnId {turn_id}: already natural clear preference answer; feedbackRequired should be false."
                        ],
                    },
                    bad_feedback,
                ]

                def sequential_chat(*args, **kwargs):
                    return json.dumps(responses.pop(0))

                self.service.chat = sequential_chat

                result = self.service.generate_feedback(request)

                self.assertGreaterEqual(result.comprehensionScore, 90)
                self.assertFalse(result.turnFeedbacks[0].feedbackRequired)
                self.assertIsNone(result.turnFeedbacks[0].nativeUnderstanding)
                self.assertIsNone(result.turnFeedbacks[0].nativeLanguageInterpretation)
                self.assertIsNone(result.turnFeedbacks[0].betterExpression)

    def test_feedback_quality_review_repairs_good_response_misclassified_as_feedback_required(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 301,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I would like a small iced Americano, please.",
                }
            ],
        })
        responses = [
            {
                "comprehensionScore": 85,
                "feedbackSummary": "전체적으로 의도를 잘 전달했습니다.",
                "turnFeedbacks": [
                    {
                        "turnId": 301,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 작은 아이스 아메리카노를 주문하고 싶다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '작은 아이스 아메리카노를 주문하고 싶어요'처럼 들려요.",
                        "betterExpression": "I'd like a small iced Americano, please. 이렇게 말하면 관사가 자연스럽게 들어갑니다.",
                    }
                ],
            },
            {
                "pass": False,
                "issues": [
                    "The user utterance is already natural, so feedbackRequired should be false.",
                    "betterExpression claims to add an article that already exists in the user's utterance.",
                ],
            },
            {
                "comprehensionScore": 95,
                "feedbackSummary": "음료 종류와 옵션을 자연스럽고 공손하게 전달했습니다.",
                "turnFeedbacks": [
                    {
                        "turnId": 301,
                        "feedbackRequired": False,
                        "nativeUnderstanding": None,
                        "nativeLanguageInterpretation": None,
                        "betterExpression": None,
                    }
                ],
            },
        ]
        systems = []

        def sequential_chat(system, *args, **kwargs):
            systems.append(system)
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(len(systems), 3)
        self.assertIn("quality reviewer", systems[1])
        self.assertIn("repair", systems[2].lower())
        self.assertEqual(result.comprehensionScore, 95)
        self.assertFalse(result.turnFeedbacks[0].feedbackRequired)
        self.assertIsNone(result.turnFeedbacks[0].nativeUnderstanding)
        self.assertIsNone(result.turnFeedbacks[0].nativeLanguageInterpretation)
        self.assertIsNone(result.turnFeedbacks[0].betterExpression)

    def test_feedback_fallback_handles_failed_repair_for_good_response(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 301,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I would like a small iced Americano, please.",
                }
            ],
        })
        bad_feedback = {
            "comprehensionScore": 85,
            "feedbackSummary": "관사 사용에 주의해 보세요.",
            "turnFeedbacks": [
                {
                    "turnId": 301,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "외국인은 '작은 아이스 아메리카노를 주문하고 싶다'고 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '작은 아이스 아메리카노를 주문하고 싶다'처럼 들려요.",
                    "betterExpression": "I'd like a small iced Americano, please. 관사가 자연스럽게 들어갑니다.",
                }
            ],
        }
        responses = [
            bad_feedback,
            {
                "pass": False,
                "issues": ["The user utterance is already natural, so feedbackRequired should be false."],
            },
            bad_feedback,
        ]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertGreaterEqual(result.comprehensionScore, 90)
        self.assertIn("자연스럽고 명확하게", result.feedbackSummary)
        self.assertFalse(result.turnFeedbacks[0].feedbackRequired)
        self.assertIsNone(result.turnFeedbacks[0].nativeUnderstanding)
        self.assertIsNone(result.turnFeedbacks[0].nativeLanguageInterpretation)
        self.assertIsNone(result.turnFeedbacks[0].betterExpression)

    def test_feedback_fallback_overrides_reviewer_pass_for_likely_good_response(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 301,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I would like a small iced Americano, please.",
                }
            ],
        })
        feedback = {
            "comprehensionScore": 85,
            "feedbackSummary": "관사 사용에 주의해 보세요.",
            "turnFeedbacks": [
                {
                    "turnId": 301,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "외국인은 사용자가 작은 아이스 아메리카노를 주문하고 싶다고 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '작은 아이스 아메리카노를 주문하고 싶다'처럼 들려요.",
                    "betterExpression": "I'd like a small iced Americano, please. 관사가 자연스럽게 들어갑니다.",
                }
            ],
        }
        responses = [
            feedback,
            {"pass": True, "issues": []},
            feedback,
        ]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertFalse(result.turnFeedbacks[0].feedbackRequired)
        self.assertIsNone(result.turnFeedbacks[0].nativeUnderstanding)
        self.assertIsNone(result.turnFeedbacks[0].nativeLanguageInterpretation)
        self.assertIsNone(result.turnFeedbacks[0].betterExpression)

    def test_feedback_summary_does_not_sound_corrective_when_all_turns_are_good(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 311,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I would like a small iced Americano, please.",
                }
            ],
        })
        self.service.chat = lambda *args, **kwargs: json.dumps({
            "comprehensionScore": 90,
            "feedbackSummary": "주문이 명확하게 전달되었습니다. 더 자연스럽게 표현해보세요.",
            "turnFeedbacks": [
                {
                    "turnId": 311,
                    "feedbackRequired": False,
                    "nativeUnderstanding": None,
                    "nativeLanguageInterpretation": None,
                    "betterExpression": None,
                }
            ],
        })

        result = self.service.generate_feedback(request)

        self.assertIn("자연스럽고 명확하게", result.feedbackSummary)
        self.assertNotIn("더 자연스럽게", result.feedbackSummary)

    def test_feedback_fallback_normalizes_known_refusal_format_after_failed_repair(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 302,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I do not want to order anything.",
                }
            ],
        })
        bad_feedback = {
            "comprehensionScore": 39,
            "feedbackSummary": "주문하려는 의도가 전혀 전달되지 않았습니다.",
            "turnFeedbacks": [
                {
                    "turnId": 302,
                    "feedbackRequired": True,
                    "nativeUnderstanding": "외국인은 '아무것도 주문하지 않겠다' 라고 이해했어요.",
                    "nativeLanguageInterpretation": "한국어로 비유하자면, '주문 자체를 거절하는 것처럼 들려요.'",
                    "betterExpression": "I'd like to order a coffee, please. 이렇게 말하면 원하는 음료를 명확히 전달할 수 있어요.",
                }
            ],
        }
        responses = [bad_feedback, bad_feedback]

        def sequential_chat(*args, **kwargs):
            return json.dumps(responses.pop(0))

        self.service.chat = sequential_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(
            result.turnFeedbacks[0].nativeUnderstanding,
            "외국인은 사용자가 아무것도 주문하지 않겠다고 이해했어요.",
        )
        self.assertEqual(
            result.turnFeedbacks[0].nativeLanguageInterpretation,
            "한국어로 비유하자면, '주문 자체를 거절하는 것'처럼 들려요.",
        )

    def test_feedback_skips_quality_review_when_response_is_not_ambiguous(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
            "turns": [
                {
                    "turnId": 101,
                    "originalQuestion": "What would you like to order?",
                    "userUtterance": "I want iced americano.",
                }
            ],
        })
        calls = []

        def capture_chat(system, *args, **kwargs):
            calls.append(system)
            return json.dumps({
                "comprehensionScore": 82,
                "feedbackSummary": "의도는 전달됐지만 표현이 조금 짧습니다.",
                "turnFeedbacks": [
                    {
                        "turnId": 101,
                        "feedbackRequired": True,
                        "nativeUnderstanding": "외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.",
                        "betterExpression": "I'd like an iced Americano, please. 이렇게 말하면 더 자연스럽습니다.",
                    }
                ],
            })

        self.service.chat = capture_chat

        result = self.service.generate_feedback(request)

        self.assertEqual(len(calls), 1)
        self.assertTrue(result.turnFeedbacks[0].feedbackRequired)

    def test_feedback_prompt_contains_stable_good_response_rubric_and_plus_one_policy(self):
        prompt = self.service._feedback_system_prompt()

        self.assertIn("Domain-neutral policy", prompt)
        self.assertIn("Classification Policy", prompt)
        self.assertIn("Field Policy", prompt)
        self.assertIn("Self-check before output", prompt)
        self.assertIn("Classify each turn before writing feedback fields", prompt)
        self.assertIn("Clear preference or option answer", prompt)
        self.assertIn("Incomplete order fragment", prompt)
        self.assertIn("Generic object response", prompt)
        self.assertIn("Direct want + concrete service item response", prompt)
        self.assertIn("must be treated as a near-miss response", prompt)
        self.assertIn("Do not invent a specific service item for incomplete order fragments or generic object responses", prompt)
        self.assertIn("Stable feedback decision rubric", prompt)
        self.assertIn("85-100", prompt)
        self.assertIn("feedbackRequired=false", prompt)
        self.assertIn("Only set feedbackRequired=false when all Good Response Conditions pass", prompt)
        self.assertIn("betterExpression +1 policy", prompt)
        self.assertIn("Keep the user's original intent, vocabulary level, and sentence shape", prompt)
        self.assertIn("If the scenario goal is not achieved, comprehensionScore must be 59 or below", prompt)
        self.assertIn("Nonsense, off-topic, refusal, or vague non-answer utterances must score 0-39", prompt)
        self.assertIn("Evaluate grammar correctness, naturalness, and fluency", prompt)
        self.assertIn("Deduct points for unnatural phrasing, missing articles, awkward word order, overly literal expressions, or robotic expressions", prompt)
        self.assertIn("Do not give 100 unless the utterance is completely natural and idiomatic", prompt)
        self.assertIn("Do not evaluate capitalization, punctuation, or spelling because the input is based on spoken utterances", prompt)
        self.assertIn("feedbackSummary must be 2 short Korean sentences by default", prompt)
        self.assertIn("Never return a one-sentence feedbackSummary", prompt)
        self.assertIn("Use 3 sentences only when multiple turns share a recurring grammar or expression pattern", prompt)
        self.assertIn("Keep feedbackSummary under 120 Korean characters", prompt)
        self.assertIn("When every turn has feedbackRequired=false", prompt)
        self.assertIn("must not imply that the user needs correction", prompt)
        self.assertIn("Verify all-good sessions do not receive correction-like summary wording", prompt)
        self.assertIn("Do not repeat detailed per-turn explanations", prompt)
        self.assertIn("betterExpression must start with the English improved sentence", prompt)
        self.assertNotIn("음료를 주문할 때는 I'd like", prompt)
        self.assertIn("I want ice one", prompt)
        self.assertIn("I'd like it iced, please.", prompt)
        self.assertIn("This drink is hot, but I ordered an iced one.", prompt)
        self.assertIn("Few-shot calibration examples", prompt)
        self.assertIn("I want drink", prompt)
        self.assertIn("Can you recommend a menu?", prompt)
        self.assertIn("That's all.", prompt)
        self.assertIn("Window seat, please.", prompt)
        self.assertIn("Non-smoking room, please.", prompt)
        self.assertIn("Table for two, please.", prompt)
        self.assertIn("Preserve the user's conversational intent", prompt)
        self.assertNotIn("natural cafe order", prompt)
        self.assertNotIn("Concrete drink values include", prompt)

    def test_feedback_repair_prompt_shares_core_classification_policy(self):
        prompt = self.service._feedback_repair_system_prompt()

        self.assertIn("Classification Policy", prompt)
        self.assertIn("Incomplete order fragment", prompt)
        self.assertIn("Generic object response", prompt)
        self.assertIn("Direct want + concrete service item response", prompt)
        self.assertIn("must be treated as a near-miss response", prompt)
        self.assertIn("Do not invent a specific service item for incomplete order fragments or generic object responses", prompt)
        self.assertIn("Self-check before output", prompt)
        self.assertIn("Never return a one-sentence feedbackSummary", prompt)
        self.assertIn("Do not write nativeUnderstanding as if the listener heard the English words", prompt)
        self.assertIn("For concrete orderable responses, nativeUnderstanding must use a Korean paraphrase of the meaning", prompt)
        self.assertIn("Do not wrap the Korean paraphrase in quotation marks inside nativeUnderstanding", prompt)
        self.assertIn("Clear preference or option answer", prompt)

    def test_feedback_prompt_constrains_turn_feedback_copy_contract(self):
        prompt = self.service._feedback_system_prompt()

        self.assertIn("nativeUnderstanding must explain what the foreign listener understood", prompt)
        self.assertIn("nativeUnderstanding must start with '외국인은'", prompt)
        self.assertIn("nativeUnderstanding must end with '라고 이해했어요.'", prompt)
        self.assertIn("For incomplete fragments, nativeUnderstanding may explain that the foreign listener could not understand the missing object", prompt)
        self.assertIn("Incomplete fragments such as bare 'I want' must keep the fragment's literal sound", prompt)
        self.assertIn("nativeUnderstanding must be based only on the same turn's userUtterance", prompt)
        self.assertIn("Do not include grammar explanations, improvement directions, or evaluations in nativeUnderstanding", prompt)
        self.assertIn("nativeUnderstanding must be one Korean sentence with a concrete interpretation", prompt)
        self.assertIn("Do not quote the user's utterance in nativeUnderstanding", prompt)
        self.assertIn("Do not write nativeUnderstanding as if the listener heard the English words", prompt)
        self.assertIn("For concrete orderable responses, nativeUnderstanding must use a Korean paraphrase of the meaning", prompt)
        self.assertIn("Do not wrap the Korean paraphrase in quotation marks inside nativeUnderstanding", prompt)
        self.assertIn("Never write patterns like 외국인은 'I want coffee'라고 들었고", prompt)
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
        self.assertIn("If the exact answer is unknown, use a simple concrete English example", prompt)
        self.assertIn("a small, achievable improvement of roughly 5 to 10 points", prompt)

    def test_feedback_uses_deterministic_chat_settings(self):
        from app.models.conversation import ConversationFeedbackRequest

        request = ConversationFeedbackRequest.model_validate({
            "scenarioTitle": "카페에서 주문하기",
            "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
            "sessionResult": "SUCCESS",
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
                        "nativeUnderstanding": "외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.",
                        "nativeLanguageInterpretation": "한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.",
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
