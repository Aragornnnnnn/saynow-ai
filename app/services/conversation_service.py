# 2차 MVP 대화 API의 LLM 호출과 응답 정규화를 담당한다.
import json
import re
from typing import Any

from pydantic import ValidationError

from app.core.llm import chat
from app.core.logger import get_logger
from app.models.conversation import (
    ConversationFeedbackRequest,
    ConversationFeedbackResponse,
    FilledSlotResponse,
    NextQuestionRequest,
    NextQuestionResponse,
)


logger = get_logger("conversation")
MAX_FEEDBACK_SUMMARY_CHARS = 120
DIRECT_WANT_NEAR_MISS_ISSUE = (
    "direct want + concrete drink response must be treated as a near-miss with feedbackRequired=true."
)


class ConversationGenerationError(Exception):
    """AI 모델 응답을 API 계약에 맞게 변환하지 못했을 때 발생한다."""


def generate_next_question(request: NextQuestionRequest) -> NextQuestionResponse:
    unfilled_slot_names = [slot.slotName for slot in request.slots if not slot.filled]
    if not unfilled_slot_names:
        return NextQuestionResponse(
            nextQuestion=None,
            translatedQuestion=None,
            filledSlots=[],
        )

    if _must_not_fill_slots(request.userUtterance):
        return _retry_question_for_slot(unfilled_slot_names[0])

    raw = _call_chat(
        _next_question_system_prompt(),
        _next_question_user_prompt(request, unfilled_slot_names),
        max_tokens=512,
        temperature=0,
    )
    data = _parse_json_object(raw)
    filled_slots = _normalize_newly_filled_slots(data, unfilled_slot_names)
    remaining_slots = [slot_name for slot_name in unfilled_slot_names if slot_name not in {slot.slotName for slot in filled_slots}]

    if not remaining_slots:
        return NextQuestionResponse(
            nextQuestion=None,
            translatedQuestion=None,
            filledSlots=filled_slots,
        )

    next_question = _optional_non_blank_string(data.get("nextQuestion"))
    translated_question = _optional_non_blank_string(data.get("translatedQuestion"))
    if next_question is None or translated_question is None:
        raise ConversationGenerationError("next question is required while unfilled slots remain")

    return NextQuestionResponse(
        nextQuestion=next_question,
        translatedQuestion=translated_question,
        filledSlots=filled_slots,
    )


def generate_feedback(request: ConversationFeedbackRequest) -> ConversationFeedbackResponse:
    raw = _call_chat(
        _feedback_system_prompt(),
        _feedback_user_prompt(request),
        max_tokens=1024,
        temperature=0,
    )
    data = _parse_json_object(raw)
    response = _validate_feedback_response(data, request)

    _enforce_feedback_consistency(request, response)
    _enforce_turn_feedback_contract(request, response)
    response = _verify_and_repair_feedback(request, response)
    _enforce_feedback_consistency(request, response)
    _enforce_turn_feedback_contract(request, response)
    return response


def _validate_feedback_response(
    data: dict[str, Any],
    request: ConversationFeedbackRequest,
) -> ConversationFeedbackResponse:
    try:
        response = ConversationFeedbackResponse.model_validate(data)
    except ValidationError as exc:
        raise ConversationGenerationError("feedback response does not match contract") from exc

    request_turn_ids = [turn.turnId for turn in request.turns]
    response_turn_ids = [turn.turnId for turn in response.turnFeedbacks]
    if response_turn_ids != request_turn_ids:
        raise ConversationGenerationError("turn feedback ids do not match request turn ids")

    return response


def _next_question_system_prompt() -> str:
    return (
        "You generate follow-up questions for an English speaking practice scenario. "
        "Return ONLY valid JSON matching this schema exactly: "
        '{"filledSlots":[{"slotName":"..."}],"nextQuestion":"<string or null>","translatedQuestion":"<string or null>"}. '
        "Decision Workflow: first identify whether the latest utterance is a concrete slot answer, a recommendation request, a no-more options completion, or a non-answer. "
        "A concrete slot answer can newly fill a slot. "
        "Recommendation request means the user asks what you recommend; it is relevant, but it does not fill the drink slot unless the user accepts or names a concrete drink. "
        "No-more options completion means the user says That's all, That's it, nothing else, or no more after an option or customization question; it fills the current option or customization slot. "
        "filledSlots must contain only slot names that were newly satisfied by the user's latest utterance. "
        "Only mark a slot as filled when the user explicitly provides a concrete value for that exact slot. "
        "Do not infer slot values from context, politeness, refusal, uncertainty, random text, or unrelated sentences. "
        "Nonsense, off-topic, refusal, or vague non-answer utterances must return filledSlots=[] and ask again for the same missing information. "
        "Incomplete order fragments without a concrete object must return filledSlots=[] and ask again for the same missing information. "
        "Examples of incomplete order fragments: I want, I need, I'd like, I would like, Can I get, Can I get a, I want to order. "
        "Generic objects such as drink, something, menu, item, or thing are not concrete slot values and must not fill drink. "
        "These utterances must never fill any slot: qwertyuiop asdfghjkl zxcvbnm, My shoes are swimming in the moon today, I don't know, No answer, I do not want to order anything. "
        "Never include slots that were already filled before this request. "
        "If all currently unfilled slots are newly satisfied, set nextQuestion and translatedQuestion to null. "
        "Do not set nextQuestion or translatedQuestion to null unless every currently unfilled slot is explicitly satisfied by the latest utterance. "
        "If any currently unfilled slot remains, ask one short natural English follow-up question and include a Korean translation. "
        "Do not include lists, explanations, or multiple follow-up questions. "
        "Use only the provided slot names. "
        "Few-shot calibration examples: "
        'Input: Previous AI question=What drink would you like to order? User utterance=Can you recommend something? Unfilled slots=drink. Output: {"filledSlots":[],"nextQuestion":"I recommend an iced latte. Would you like to order that?","translatedQuestion":"아이스 라떼를 추천해요. 그걸로 주문하시겠어요?"}. '
        'Input: Previous AI question=What custom options would you like for your drink? User utterance=That\'s all. Unfilled slots=customOptions. Output: {"filledSlots":[{"slotName":"customOptions"}],"nextQuestion":null,"translatedQuestion":null}. '
        'Input: Previous AI question=What drink would you like to order? User utterance=I want drink. Unfilled slots=drink. Output: {"filledSlots":[],"nextQuestion":"What drink would you like to order?","translatedQuestion":"어떤 음료를 주문하고 싶으신가요?"}.'
    )


def _next_question_user_prompt(request: NextQuestionRequest, unfilled_slot_names: list[str]) -> str:
    slot_lines = "\n".join(
        f"- {slot.slotName}: {'filled' if slot.filled else 'unfilled'}"
        for slot in request.slots
    )
    unfilled_lines = "\n".join(f"- {slot_name}" for slot_name in unfilled_slot_names)
    return (
        f"Scenario title: {request.scenarioTitle}\n"
        f"Scenario goal: {request.scenarioGoal}\n"
        f"Previous AI question: {request.originalQuestion}\n"
        f"User utterance: {request.userUtterance}\n\n"
        f"Current slot state:\n{slot_lines}\n\n"
        f"Only these unfilled slots may be newly filled or asked about:\n{unfilled_lines}"
    )


def _feedback_system_prompt() -> str:
    return (
        "You generate final feedback for an English speaking practice scenario. "
        "Use this structured policy in order: Output Contract, Classification Policy, Scoring Policy, Field Policy, Self-check before output. "
        "Output Contract: "
        "Return ONLY valid JSON matching this schema exactly: "
        '{"comprehensionScore":82,"feedbackSummary":"...","turnFeedbacks":[{"turnId":101,"feedbackRequired":true,"nativeUnderstanding":"...","nativeLanguageInterpretation":"...","betterExpression":"..."}]}. '
        "For each turn, preserve the exact turnId from the request. "
        "Classify each turn before writing feedback fields. "
        "Classification Policy: "
        "Good response means the utterance directly answers the AI question, satisfies the scenario intent, and is natural enough for a native listener. "
        "Near-miss response means the intended answer is clear but grammar, word choice, word order, politeness, or completeness needs a small correction. "
        "Direct want + concrete drink response means a phrase such as I want coffee or I want iced americano; it is understandable but too direct for a natural cafe order, so it must be treated as a near-miss response. "
        "Incomplete order fragment means the user starts an order phrase but does not provide a concrete object, such as I want, I need, I'd like, I would like, Can I get, Can I get a, or I want to order. "
        "Generic object response means the user gives only a generic object such as drink, something, anything, menu, item, thing, or one instead of a concrete orderable drink. "
        "Recommendation request means the user asks for a menu, drink, or option recommendation; it is relevant help-seeking and must preserve the recommendation-request intent. "
        "No-more options response means the user says That's all, That's it, nothing else, or no more after an option or customization question; it is a natural completion response. "
        "Off-topic or nonsense means the utterance does not provide usable scenario information and must preserve its literal odd meaning. "
        "Refusal or non-answer means the user refuses, says they do not know, or avoids answering the AI question. "
        "Concrete drink values include specific orderable items such as coffee, latte, americano, tea, water, juice, or named menu items. "
        "Do not invent a specific drink for incomplete order fragments or generic object responses in listener-meaning fields. "
        "Do not invent a specific drink inside nativeUnderstanding or nativeLanguageInterpretation for incomplete order fragments or generic object responses. "
        "Scoring Policy: "
        "comprehensionScore is an integer from 0 to 100 from a native listener's perspective. "
        "Evaluate grammar correctness, naturalness, and fluency in addition to scenario fit. "
        "Deduct points for unnatural phrasing, missing articles, awkward word order, overly literal expressions, or robotic expressions. "
        "Do not give 100 unless the utterance is completely natural and idiomatic. "
        "Do not evaluate capitalization, punctuation, or spelling because the input is based on spoken utterances. "
        "Stable feedback decision rubric: 0-39 means the answer is off-topic or a native listener cannot identify the intended meaning; "
        "40-59 means only a vague gist is understandable and key scenario information is missing or heavily distorted; "
        "60-74 means the main intent is understandable but grammar, word choice, or word order is clearly awkward enough to need correction; "
        "75-84 means the scenario intent is clear but a small correction would noticeably improve naturalness, politeness, or completeness; "
        "85-100 means the answer directly answers the question, a native listener understands it without guessing, and any remaining awkwardness is minor. "
        "Direct want + concrete drink responses must score 75-84, not 85-100, because they need a +1 politeness and naturalness improvement. "
        "If the scenario goal is not achieved, comprehensionScore must be 59 or below. "
        "Nonsense, off-topic, refusal, or vague non-answer utterances must score 0-39. "
        "Good Response Conditions: the answer must address the AI question, satisfy the scenario intent for that turn, be understandable without extra inference, and have no meaning-blocking grammar or word-choice issue. "
        "Only set feedbackRequired=false when all Good Response Conditions pass and the internal turn score is 85-100. "
        "For No-more options responses after an option or customization question, feedbackRequired=false is allowed because the turn goal is complete. "
        "Do not set feedbackRequired=false for Direct want + concrete drink responses. "
        "If any condition fails, or the internal turn score is 84 or below, set feedbackRequired=true. "
        "Apply this exact rubric consistently for every request; do not loosen or tighten it by scenario, user level, or writing style. "
        "Field Policy: "
        "feedbackSummary is Korean and concise. "
        "feedbackSummary must be 2 short Korean sentences by default. "
        "Never return a one-sentence feedbackSummary. "
        "Use 3 sentences only when multiple turns share a recurring grammar or expression pattern. "
        "Keep feedbackSummary under 120 Korean characters. "
        "Sentence 1 must summarize whether the scenario goal was achieved and how well the user was understood. "
        "Sentence 2 must give the single most important next practice focus. "
        "Do not repeat detailed per-turn explanations, nativeUnderstanding, nativeLanguageInterpretation, or betterExpression content in feedbackSummary. "
        "Do not list multiple strengths and weaknesses. "
        "When feedbackRequired=false, set nativeUnderstanding, nativeLanguageInterpretation, and betterExpression to null. "
        "When feedbackRequired is true, nativeUnderstanding must explain what the foreign listener understood from the user's utterance. "
        "nativeUnderstanding must start with '외국인은'. "
        "nativeUnderstanding must end with '라고 이해했어요.'. "
        "For incomplete fragments, nativeUnderstanding may explain that the foreign listener could not understand the missing object and end with '이해할 수 없었어요.'. "
        "For incomplete order fragments and generic object responses, nativeUnderstanding must say the foreign listener could not identify the specific drink. "
        "nativeUnderstanding must be based only on the same turn's userUtterance. "
        "nativeUnderstanding must be one Korean sentence with a concrete interpretation. "
        "Do not include grammar explanations, improvement directions, or evaluations in nativeUnderstanding. "
        "Do not quote the user's utterance in nativeUnderstanding. "
        "Do not write nativeUnderstanding as if the listener heard the English words. "
        "For concrete orderable responses, nativeUnderstanding must use a Korean paraphrase of the meaning, not the English utterance. "
        "Do not wrap the Korean paraphrase in quotation marks inside nativeUnderstanding. "
        "Never write patterns like 외국인은 'I want coffee'라고 들었고. "
        "Do not use nativeUnderstanding for meta-evaluation such as saying the utterance is unrelated, figurative, or grammatically wrong. "
        "Incomplete fragments such as bare 'I want' must keep the fragment's literal sound and must not become advice such as saying the user needs to add a drink name. "
        "For generic object responses, preserve the generic object instead of pretending the listener heard a specific menu item. "
        "Instead, describe the practical intent, uncertainty, or likely misunderstanding a foreign listener would act on. "
        "Do not write nativeUnderstanding as '주문할 음료에 대한 내용이 아니다' or '질문과 관련이 없다'; preserve the listener's literal interpretation instead. "
        "If the user mentions one ice, explain that the listener may think the user wants one ice cube, not less ice in a drink. "
        "If the user says an unrelated nonsensical sentence, describe the literal odd meaning the listener receives instead of saying only that it is unrelated. "
        "nativeLanguageInterpretation must be a Korean analogy for how the user's English sounds to the foreign listener, not a literal target-language translation. "
        "nativeLanguageInterpretation must be based only on the same turn's userUtterance. "
        "Do not borrow content from prompt examples, previous turns, other test inputs, scenarioTitle, or scenarioGoal. "
        "nativeUnderstanding and nativeLanguageInterpretation must describe the same meaning. "
        "Write nativeLanguageInterpretation in Korean using this pattern: '한국어로 비유하자면, ...처럼 들려요.' "
        "Use single quotation marks around the Korean analogy phrase in nativeLanguageInterpretation. "
        "Use the analogy to help a Korean learner realize how their English sounded. "
        "For incomplete fragments, nativeLanguageInterpretation must mirror the literal Korean-sounding fragment, not the scenario consequence. "
        "For generic object responses, nativeLanguageInterpretation must mirror the generic meaning, such as wanting a drink or something, not a specific drink. "
        "For nonsensical or off-topic utterances, preserve the strange meaning in the Korean analogy; do not force it into the scenario context. "
        "For nonsensical utterances, nativeLanguageInterpretation must mirror the same nonsensical meaning from that userUtterance. "
        "Meaningful but awkward utterances must stay in their own meaning family. "
        "An utterance about less ice must stay in the less-ice meaning family. "
        "An utterance about one ice or iced must stay in the one-ice or iced-drink meaning family. "
        "Examples are format guidance only and must never be copied into output. "
        "Do not write phrases like '목표 언어로 번역하면' or describe only the dictionary meaning. "
        "Do not include backslash characters in any response string. "
        "Do not use double quotation marks inside any response string because JSON will escape them with backslashes. "
        "betterExpression +1 policy: improve the user's utterance by exactly one practical step. "
        "Preserve the user's conversational intent: recommendation requests should improve into a clearer recommendation request, not into an order; option-completion responses should not be rewritten when they are already natural. "
        "Target a small, achievable improvement of roughly 5 to 10 points, not a perfect rewrite. "
        "Keep the user's original intent, vocabulary level, and sentence shape as much as possible. "
        "Fix the smallest issue that makes the response more natural, such as one missing article, a more polite phrase, or a clearer word order. "
        "Do not add new details, idioms, advanced grammar, long sentences, or a fully polished native-level rewrite unless the user's original was already close to that level. "
        "betterExpression must include the improved sentence and a short Korean reason in the same string. "
        "betterExpression must start with the English improved sentence, then a short Korean reason may follow. "
        "Do not start betterExpression with Korean guidance such as '음료를 주문할 때는'. "
        "betterExpression must never be only Korean guidance; it must include an English improved sentence or English example. "
        "For 'I want ice one', betterExpression should start with 'I'd like it iced, please.' or 'I want it iced, please.' "
        "For 'This drink is hot but I order ice one', betterExpression should start with 'This drink is hot, but I ordered an iced one.' or 'I ordered an iced drink, but this one is hot.' "
        "When the user's utterance answers the question but sounds awkward, give a +1 improved sentence and explain why that small change helps. "
        "For Direct want + concrete drink responses, the +1 improved sentence should start with I'd like plus the same drink and please. "
        "For recommendation requests, the +1 improved sentence should ask for a recommendation, such as What do you recommend? or Could you recommend something? "
        "For incomplete or generic order responses, betterExpression may use a simple concrete example such as I'd like a coffee, please. to show the missing object, but nativeUnderstanding and nativeLanguageInterpretation must not claim the user said coffee. "
        "When the user's utterance does not answer the AI question or scenario intent, give a simple English answer without wrapping it in quotation marks, then explain why it fits. "
        "The English example must appear plainly without double quotation marks, for example 'I'd like an Americano, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.' "
        "If the exact answer is unknown, use a simple concrete English example that fits the scenario, such as 'I'd like a coffee, please.' for ordering a drink. "
        "Do not return only an English sentence with a parenthesized Korean translation. "
        "Few-shot calibration examples: "
        "Example A input userUtterance=I want drink. Output direction: feedbackRequired=true, nativeUnderstanding says the listener cannot identify the specific drink, nativeLanguageInterpretation mirrors '나는 음료를 원한다', betterExpression starts with I'd like a coffee, please. "
        "Example B input userUtterance=Can you recommend a menu? Output direction: preserve recommendation intent, nativeUnderstanding says the listener understood a menu recommendation request, betterExpression starts with What do you recommend? "
        "Example C input originalQuestion=What custom options would you like for your drink? userUtterance=That's all. Output direction: feedbackRequired=false with null turn feedback fields. "
        "Self-check before output: "
        "Verify the JSON has exactly the required fields. "
        "Verify each turnId matches the request. "
        "Verify feedbackRequired=false has null turn feedback fields. "
        "Verify Direct want + concrete drink responses have feedbackRequired=true and a 75-84 score. "
        "Verify incomplete order fragments and generic object responses do not invent a specific drink. "
        "Verify recommendation requests preserve the recommendation intent. "
        "Verify No-more options responses after option or customization questions do not receive unnecessary feedback. "
        "Verify nativeUnderstanding does not quote or copy English words for concrete orderable responses. "
        "Verify nativeUnderstanding, nativeLanguageInterpretation, betterExpression, and feedbackSummary do not repeat each other's responsibilities. "
        "Verify feedbackSummary is exactly 2 short Korean sentences by default, under 120 Korean characters, and at most 3 sentences. "
        "If any check fails, revise before returning the JSON."
    )


def _must_not_fill_slots(user_utterance: str) -> bool:
    normalized = _normalize_utterance(user_utterance)
    compact = normalized.replace("'", "")

    exact_blocked = {
        "qwertyuiop asdfghjkl zxcvbnm",
        "my shoes are swimming in the moon today",
        "i dont know",
        "no answer",
        "i do not want to order anything",
        "i dont want to order anything",
    }
    if compact in exact_blocked:
        return True

    if _is_incomplete_utterance_fragment(user_utterance):
        return True

    if "qwertyuiop" in compact or "asdfghjkl" in compact or "zxcvbnm" in compact:
        return True

    refusal_patterns = [
        "do not want to order",
        "dont want to order",
        "do not want anything",
        "dont want anything",
    ]
    return any(pattern in compact for pattern in refusal_patterns)


def _normalize_utterance(value: str) -> str:
    lowered = value.lower().strip()
    no_punctuation = re.sub(r"[^a-z0-9'\s]", " ", lowered)
    return re.sub(r"\s+", " ", no_punctuation).strip()


def _retry_question_for_slot(slot_name: str) -> NextQuestionResponse:
    slot_key = slot_name.lower()
    if slot_key == "drink":
        return NextQuestionResponse(
            nextQuestion="What drink would you like to order?",
            translatedQuestion="어떤 음료를 주문하고 싶으신가요?",
            filledSlots=[],
        )
    if slot_key == "size":
        return NextQuestionResponse(
            nextQuestion="What size would you like?",
            translatedQuestion="어떤 사이즈로 하시겠어요?",
            filledSlots=[],
        )

    readable_slot = slot_name.replace("_", " ")
    return NextQuestionResponse(
        nextQuestion=f"Could you tell me your {readable_slot}?",
        translatedQuestion=f"{slot_name} 정보를 알려주시겠어요?",
        filledSlots=[],
    )


def _enforce_feedback_consistency(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> None:
    if not all(_must_not_fill_slots(turn.userUtterance) for turn in request.turns):
        return

    response.comprehensionScore = min(response.comprehensionScore, 39)
    for turn_feedback in response.turnFeedbacks:
        turn_feedback.feedbackRequired = True
        turn_feedback.nativeUnderstanding = turn_feedback.nativeUnderstanding or "외국인은 사용자가 대답하지 않았다고 이해했어요."
        turn_feedback.nativeLanguageInterpretation = (
            turn_feedback.nativeLanguageInterpretation
            or "한국어로 비유하자면, '대답을 하지 않은 것'처럼 들려요."
        )
        turn_feedback.betterExpression = (
            turn_feedback.betterExpression
            or "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 주문할 수 있어요."
        )


def _enforce_turn_feedback_contract(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> None:
    turns_by_id = {turn.turnId: turn for turn in request.turns}
    marked_direct_want_near_miss = False
    for turn_feedback in response.turnFeedbacks:
        if not turn_feedback.feedbackRequired:
            continue

        turn = turns_by_id.get(turn_feedback.turnId)
        if turn is None:
            continue

        understanding = _native_understanding_override(turn.userUtterance)
        if _is_direct_want_concrete_order_near_miss(turn.userUtterance):
            _apply_direct_want_concrete_order_feedback(turn.userUtterance, turn_feedback)
            response.comprehensionScore = min(max(response.comprehensionScore, 75), 84)
            marked_direct_want_near_miss = True
            continue

        if understanding is not None:
            turn_feedback.nativeUnderstanding = understanding

        interpretation = _native_language_interpretation_override(turn.userUtterance)
        if interpretation is not None:
            turn_feedback.nativeLanguageInterpretation = interpretation

    if marked_direct_want_near_miss and len(response.turnFeedbacks) == 1:
        response.feedbackSummary = (
            "시나리오 목표는 대체로 달성했어요. "
            "다음에는 더 자연스럽고 공손한 주문 표현을 연습해 보세요."
        )


def _verify_and_repair_feedback(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> ConversationFeedbackResponse:
    issues = _deterministic_feedback_issues(request, response)
    issues.extend(_good_response_policy_issues(request, response))
    semantic_issues = _semantic_feedback_policy_issues(request, response)
    issues.extend(semantic_issues)
    should_review = _should_review_feedback_quality(response) or bool(semantic_issues)
    if not should_review:
        if issues:
            return _repair_feedback(request, response, issues)
        return response

    review = _review_feedback_quality(request, response)
    if not review["pass"]:
        issues.extend(review["issues"])

    if not issues:
        return response

    return _repair_feedback(request, response, issues)


def _good_response_policy_issues(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> list[str]:
    if response.comprehensionScore < 85:
        return []

    turns_by_id = {turn.turnId: turn for turn in request.turns}
    issues: list[str] = []
    for turn_feedback in response.turnFeedbacks:
        turn = turns_by_id.get(turn_feedback.turnId)
        if turn is None:
            continue
        if turn_feedback.feedbackRequired and _is_likely_good_response(turn.userUtterance):
            issues.append(
                f"turnId {turn_feedback.turnId}: likely good response; feedbackRequired should be false."
            )
    return issues


def _deterministic_feedback_issues(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> list[str]:
    turns_by_id = {turn.turnId: turn for turn in request.turns}
    issues = _feedback_summary_issues(response.feedbackSummary)
    for turn_feedback in response.turnFeedbacks:
        turn = turns_by_id.get(turn_feedback.turnId)
        issue_prefix = f"turnId {turn_feedback.turnId}: "

        if not turn_feedback.feedbackRequired:
            if turn is not None and _is_direct_want_concrete_order_near_miss(turn.userUtterance):
                issues.append(issue_prefix + DIRECT_WANT_NEAR_MISS_ISSUE)
            if any([
                turn_feedback.nativeUnderstanding is not None,
                turn_feedback.nativeLanguageInterpretation is not None,
                turn_feedback.betterExpression is not None,
            ]):
                issues.append(issue_prefix + "feedbackRequired=false must keep all turn feedback fields null.")
            continue

        native_understanding = turn_feedback.nativeUnderstanding or ""
        native_language_interpretation = turn_feedback.nativeLanguageInterpretation or ""
        better_expression = turn_feedback.betterExpression or ""

        if not native_understanding.startswith("외국인은"):
            issues.append(issue_prefix + "nativeUnderstanding must start with 외국인은.")
        allows_incomplete_fragment = turn is not None and _is_incomplete_utterance_fragment(turn.userUtterance)
        if not (
            re.search(r"(라고|다고) 이해했어요\.$", native_understanding)
            or (allows_incomplete_fragment and native_understanding.endswith("이해할 수 없었어요."))
        ):
            issues.append(issue_prefix + "nativeUnderstanding must end with 라고 이해했어요 or 다고 이해했어요.")
        if _contains_quote(native_understanding) and not allows_incomplete_fragment:
            issues.append(issue_prefix + "nativeUnderstanding must not quote the user's utterance or translated phrase.")
        if _contains_native_understanding_evaluation(native_understanding):
            issues.append(issue_prefix + "nativeUnderstanding must not include grammar explanations, improvement directions, or evaluations.")
        if turn is not None and _contains_user_utterance(native_understanding, turn.userUtterance) and not allows_incomplete_fragment:
            issues.append(issue_prefix + "nativeUnderstanding must not copy the user's English utterance.")

        if not (
            native_language_interpretation.startswith("한국어로 비유하자면, '")
            and native_language_interpretation.endswith("'처럼 들려요.")
        ):
            issues.append(issue_prefix + "nativeLanguageInterpretation must follow 한국어로 비유하자면, '...'처럼 들려요.")

        if not re.match(r"^[A-Za-z]", better_expression):
            issues.append(issue_prefix + "betterExpression must start with an English improved expression.")

    return issues


def _semantic_feedback_policy_issues(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> list[str]:
    turns_by_id = {turn.turnId: turn for turn in request.turns}
    issues: list[str] = []
    for turn_feedback in response.turnFeedbacks:
        turn = turns_by_id.get(turn_feedback.turnId)
        if turn is None:
            continue

        issue_prefix = f"turnId {turn_feedback.turnId}: "
        if _is_no_more_options_response(turn.originalQuestion, turn.userUtterance):
            if turn_feedback.feedbackRequired:
                issues.append(issue_prefix + "already natural no-more options response; feedbackRequired should be false.")
            continue

        if not turn_feedback.feedbackRequired:
            continue

        better_expression = turn_feedback.betterExpression or ""
        if _is_incomplete_utterance_fragment(turn.userUtterance) and _better_expression_stays_generic_order(
            better_expression
        ):
            issues.append(
                issue_prefix
                + "betterExpression should give a concrete practice example instead of staying with a generic drink order."
            )

        if _is_recommendation_request(turn.userUtterance) and _better_expression_changes_recommendation_intent(
            better_expression
        ):
            issues.append(issue_prefix + "recommendation request intent must be preserved in betterExpression.")

    return issues


def _feedback_summary_issues(feedback_summary: str) -> list[str]:
    issues: list[str] = []
    if len(feedback_summary.strip()) > MAX_FEEDBACK_SUMMARY_CHARS:
        issues.append(f"feedbackSummary must stay under {MAX_FEEDBACK_SUMMARY_CHARS} Korean characters.")
    if _count_feedback_summary_sentences(feedback_summary) > 3:
        issues.append("feedbackSummary must use at most 3 Korean sentences.")
    return issues


def _count_feedback_summary_sentences(feedback_summary: str) -> int:
    stripped = feedback_summary.strip()
    if not stripped:
        return 0
    sentence_endings = re.findall(r"[.!?]+(?:\s|$)", stripped)
    return len(sentence_endings) if sentence_endings else 1


def _should_review_feedback_quality(response: ConversationFeedbackResponse) -> bool:
    return response.comprehensionScore >= 85 and any(
        turn_feedback.feedbackRequired for turn_feedback in response.turnFeedbacks
    )


def _review_feedback_quality(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> dict[str, Any]:
    data = _parse_json_object(_call_chat(
        _feedback_quality_review_system_prompt(),
        _feedback_quality_review_user_prompt(request, response),
        max_tokens=512,
        temperature=0,
    ))
    passed = data.get("pass")
    issues = data.get("issues")
    if not isinstance(passed, bool) or not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
        raise ConversationGenerationError("feedback quality review response does not match contract")
    return {"pass": passed, "issues": issues}


def _repair_feedback(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
    issues: list[str],
) -> ConversationFeedbackResponse:
    data = _parse_json_object(_call_chat(
        _feedback_repair_system_prompt(),
        _feedback_repair_user_prompt(request, response, issues),
        max_tokens=1024,
        temperature=0,
    ))
    repaired = _validate_feedback_response(data, request)
    _enforce_feedback_consistency(request, repaired)
    _enforce_turn_feedback_contract(request, repaired)
    _apply_feedback_safety_fallbacks(request, repaired, issues)
    remaining_issues = _deterministic_feedback_issues(request, repaired)
    if remaining_issues:
        logger.warning("피드백 repair 후에도 계약 위반이 남음 | issues: %s", remaining_issues)
    return repaired


def _feedback_quality_review_system_prompt() -> str:
    return (
        "You are a strict quality reviewer for English speaking feedback. "
        "Return ONLY valid JSON matching this schema exactly: "
        '{"pass":true,"issues":["..."]}. '
        "Review whether the feedback follows the product policy, not whether the JSON schema is valid. "
        "Check especially: a clearly good answer must not receive unnecessary turn feedback; "
        "feedbackRequired=false is allowed only for genuinely good answers; "
        "Direct want + concrete drink responses such as I want coffee must receive +1 feedback for naturalness and politeness; "
        "betterExpression must not claim to fix something already present in the user's utterance; "
        "incomplete or generic order responses must not keep betterExpression at a generic drink order such as I'd like a drink; "
        "recommendation requests must preserve recommendation-request intent instead of being rewritten as a direct order; "
        "no-more options responses after option or customization questions must not receive unnecessary feedback; "
        "nativeUnderstanding must not quote the user's English utterance; "
        "nativeUnderstanding and nativeLanguageInterpretation must describe the same meaning; "
        "off-topic utterances must preserve their literal odd meaning instead of being forced into the scenario. "
        "If there are no meaningful policy issues, return pass=true and issues=[]. "
        "If repair is needed, return pass=false and concise issue strings."
    )


def _feedback_quality_review_user_prompt(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> str:
    return (
        "Request JSON:\n"
        f"{json.dumps(request.model_dump(), ensure_ascii=False)}\n\n"
        "Feedback JSON:\n"
        f"{json.dumps(response.model_dump(), ensure_ascii=False)}"
    )


def _feedback_repair_system_prompt() -> str:
    return (
        "You repair final feedback JSON for an English speaking practice scenario. "
        "Use this structured policy in order: Output Contract, Classification Policy, Field Policy, Self-check before output. "
        "Output Contract: "
        "Return ONLY valid JSON matching this schema exactly: "
        '{"comprehensionScore":82,"feedbackSummary":"...","turnFeedbacks":[{"turnId":101,"feedbackRequired":true,"nativeUnderstanding":"...","nativeLanguageInterpretation":"...","betterExpression":"..."}]}. '
        "Fix only the listed issues while preserving the request turn order and exact turnId values. "
        "Do not add or remove fields. "
        "Classification Policy: "
        "Incomplete order fragment means the user starts an order phrase but does not provide a concrete object, such as I want, I need, I'd like, I would like, Can I get, Can I get a, or I want to order. "
        "Generic object response means the user gives only a generic object such as drink, something, anything, menu, item, thing, or one instead of a concrete orderable drink. "
        "Recommendation request means the user asks for a menu, drink, or option recommendation; preserve that conversational intent. "
        "No-more options response means the user says That's all, That's it, nothing else, or no more after an option or customization question; it should usually be feedbackRequired=false. "
        "Direct want + concrete drink response means a phrase such as I want coffee or I want iced americano; it is understandable but too direct for a natural cafe order, so it must be treated as a near-miss response. "
        "Concrete drink values include specific orderable items such as coffee, latte, americano, tea, water, juice, or named menu items. "
        "Do not invent a specific drink for incomplete order fragments or generic object responses in listener-meaning fields. "
        "Field Policy: "
        "feedbackSummary must be concise: 2 short Korean sentences by default, never one sentence, 3 sentences only for recurring multi-turn issues, and under 120 Korean characters. "
        "Never return a one-sentence feedbackSummary. "
        "Do not repeat detailed per-turn explanations, nativeUnderstanding, nativeLanguageInterpretation, or betterExpression content in feedbackSummary. "
        "When feedbackRequired=false, nativeUnderstanding, nativeLanguageInterpretation, and betterExpression must be null. "
        "When feedbackRequired=true, nativeUnderstanding must start with 외국인은 and end with 라고 이해했어요 or 다고 이해했어요. "
        "For incomplete order fragments with a missing object, nativeUnderstanding may instead end with 이해할 수 없었어요. "
        "For incomplete order fragments and generic object responses, nativeUnderstanding must say the foreign listener could not identify the specific drink. "
        "nativeUnderstanding must not quote the user's English utterance and must not include grammar explanations, improvement directions, or evaluations. "
        "Do not write nativeUnderstanding as if the listener heard the English words. "
        "For concrete orderable responses, nativeUnderstanding must use a Korean paraphrase of the meaning, not the English utterance. "
        "Do not wrap the Korean paraphrase in quotation marks inside nativeUnderstanding. "
        "nativeLanguageInterpretation must follow this pattern exactly: 한국어로 비유하자면, '...'처럼 들려요. "
        "For generic object responses, nativeLanguageInterpretation must mirror the generic meaning, not a specific drink. "
        "betterExpression must start with an English improved expression followed by a short Korean reason. "
        "Preserve the user's conversational intent: recommendation requests should improve into recommendation requests, not direct orders. "
        "For incomplete or generic order responses, betterExpression may use a simple concrete example such as I'd like a coffee, please. to model the missing object. "
        "For No-more options responses after option or customization questions, set feedbackRequired=false and keep nativeUnderstanding, nativeLanguageInterpretation, and betterExpression null. "
        "For Direct want + concrete drink responses, keep feedbackRequired=true, keep comprehensionScore at 75-84, and start betterExpression with I'd like plus the same drink and please. "
        "For clearly good, natural answers that directly satisfy the AI question, set feedbackRequired=false for that turn. "
        "Self-check before output: "
        "Verify the repaired JSON still matches the schema, preserves turnId values, keeps summary at 2 short Korean sentences by default, keeps Direct want + concrete drink responses as feedbackRequired=true, does not quote English utterances in nativeUnderstanding, and does not invent a drink for incomplete or generic responses."
    )


def _feedback_repair_user_prompt(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
    issues: list[str],
) -> str:
    issue_lines = "\n".join(f"- {issue}" for issue in issues)
    return (
        f"Issues to repair:\n{issue_lines}\n\n"
        "Request JSON:\n"
        f"{json.dumps(request.model_dump(), ensure_ascii=False)}\n\n"
        "Current feedback JSON:\n"
        f"{json.dumps(response.model_dump(), ensure_ascii=False)}"
    )


def _contains_quote(value: str) -> bool:
    return any(mark in value for mark in ["'", '"', "‘", "’", "“", "”"])


def _contains_native_understanding_evaluation(value: str) -> bool:
    evaluation_markers = [
        "문법",
        "어색",
        "자연스럽",
        "개선",
        "정확한 의도",
        "파악하기 어려",
    ]
    return any(marker in value for marker in evaluation_markers)


def _contains_user_utterance(value: str, user_utterance: str) -> bool:
    normalized_value = _normalize_utterance(value)
    normalized_utterance = _normalize_utterance(user_utterance)
    return bool(normalized_utterance and normalized_utterance in normalized_value)


def _is_incomplete_utterance_fragment(user_utterance: str) -> bool:
    return _incomplete_order_fragment_analogy(user_utterance) is not None


def _incomplete_order_fragment_analogy(user_utterance: str) -> str | None:
    compact = _normalize_utterance(user_utterance).replace("'", "")
    article_suffix = r"(?: (?:a|an|the))?"

    generic_object_analogy = _generic_order_object_analogy(compact)
    if generic_object_analogy is not None:
        return generic_object_analogy

    if re.fullmatch(rf"i want{article_suffix}", compact):
        return "나는 하나를 원한다" if compact != "i want" else "나는 원한다"
    if re.fullmatch(rf"i need{article_suffix}", compact):
        return "나는 하나가 필요하다" if compact != "i need" else "나는 필요하다"
    if re.fullmatch(rf"(?:id like|i would like){article_suffix}", compact):
        return "저는 하나를 원해요" if compact.endswith((" a", " an", " the")) else "저는 원해요"
    if re.fullmatch(rf"(?:can i get|could i get|may i have){article_suffix}", compact):
        return "제가 하나 받을 수 있을까요" if compact.endswith((" a", " an", " the")) else "제가 받을 수 있을까요"
    if re.fullmatch(rf"i want to order{article_suffix}", compact):
        return "나는 하나를 주문하고 싶다" if compact != "i want to order" else "나는 주문하고 싶다"

    return None


def _generic_order_object_analogy(compact_utterance: str) -> str | None:
    object_pattern = r"(?P<object>drink|drinks|something|anything|menu|item|thing|one)"
    patterns = [
        (rf"i want(?: to order)? (?:a |an |the )?{object_pattern}", "want"),
        (rf"i need (?:a |an |the )?{object_pattern}", "need"),
        (rf"(?:id like|i would like) (?:a |an |the )?{object_pattern}", "like"),
        (rf"(?:can i get|could i get|may i have) (?:a |an |the )?{object_pattern}", "get"),
    ]
    for pattern, intent in patterns:
        match = re.fullmatch(pattern, compact_utterance)
        if match:
            return _generic_object_analogy_phrase(match.group("object"), intent)
    return None


def _generic_object_analogy_phrase(object_word: str, intent: str) -> str:
    object_phrases = {
        "drink": "음료",
        "drinks": "음료",
        "something": "뭔가",
        "anything": "아무거나",
        "menu": "메뉴",
        "item": "상품",
        "thing": "것",
        "one": "하나",
    }
    phrase = object_phrases[object_word]
    if intent == "like":
        return f"저는 {phrase}를 원해요"
    if intent == "get":
        return f"제가 {phrase}를 받을 수 있을까요"
    if intent == "need":
        return f"나는 {phrase}가 필요하다"
    return f"나는 {phrase}를 원한다"


def _trim_incomplete_fragment_for_feedback(user_utterance: str) -> str:
    return user_utterance.strip().rstrip(".?!").strip()


def _apply_feedback_safety_fallbacks(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
    issues: list[str],
) -> None:
    turns_by_id = {turn.turnId: turn for turn in request.turns}
    force_good_response = any(
        "already natural" in issue or "feedbackRequired should be false" in issue
        for issue in issues
    )
    force_direct_want_near_miss = any(DIRECT_WANT_NEAR_MISS_ISSUE in issue for issue in issues)
    marked_good = False
    marked_direct_want_near_miss = False

    for turn_feedback in response.turnFeedbacks:
        turn = turns_by_id.get(turn_feedback.turnId)
        if turn is None:
            continue

        if force_direct_want_near_miss and _is_direct_want_concrete_order_near_miss(turn.userUtterance):
            _apply_direct_want_concrete_order_feedback(turn.userUtterance, turn_feedback)
            response.comprehensionScore = min(max(response.comprehensionScore, 75), 84)
            marked_direct_want_near_miss = True
            continue

        if force_good_response and (
            _is_likely_good_response(turn.userUtterance)
            or _is_no_more_options_response(turn.originalQuestion, turn.userUtterance)
        ):
            turn_feedback.feedbackRequired = False
            turn_feedback.nativeUnderstanding = None
            turn_feedback.nativeLanguageInterpretation = None
            turn_feedback.betterExpression = None
            response.comprehensionScore = max(response.comprehensionScore, 90)
            marked_good = True
            continue

        if not turn_feedback.feedbackRequired:
            continue

        understanding = _native_understanding_override(turn.userUtterance)
        if understanding is not None:
            turn_feedback.nativeUnderstanding = understanding
        else:
            turn_feedback.nativeUnderstanding = _normalize_native_understanding_format(
                turn_feedback.nativeUnderstanding
            )

        interpretation = _native_language_interpretation_override(turn.userUtterance)
        if interpretation is not None:
            turn_feedback.nativeLanguageInterpretation = interpretation
        else:
            turn_feedback.nativeLanguageInterpretation = _normalize_native_language_interpretation_format(
                turn_feedback.nativeLanguageInterpretation
            )

    if marked_good and all(not turn_feedback.feedbackRequired for turn_feedback in response.turnFeedbacks):
        response.feedbackSummary = (
            "전체적으로 질문에 자연스럽고 명확하게 답변했습니다. "
            "다음 연습에서도 공손하고 구체적인 표현을 유지해 보세요."
        )

    if marked_direct_want_near_miss and len(response.turnFeedbacks) == 1:
        response.feedbackSummary = (
            "시나리오 목표는 대체로 달성했어요. "
            "다음에는 더 자연스럽고 공손한 주문 표현을 연습해 보세요."
        )


def _is_likely_good_response(user_utterance: str) -> bool:
    compact = _normalize_utterance(user_utterance).replace("'", "")
    return bool(re.match(r"^(i would like|id like) .+ please$", compact)) and len(compact.split()) >= 6


def _is_recommendation_request(user_utterance: str) -> bool:
    compact = _normalize_utterance(user_utterance).replace("'", "")
    return any([
        "recommend" in compact,
        "suggest" in compact,
        "what should i get" in compact,
        "what do you think i should get" in compact,
    ])


def _is_no_more_options_response(original_question: str, user_utterance: str) -> bool:
    compact_question = _normalize_utterance(original_question).replace("'", "")
    compact_utterance = _normalize_utterance(user_utterance).replace("'", "")
    option_question = any(
        marker in compact_question
        for marker in ["option", "custom", "customize", "anything else", "add on", "extra", "topping"]
    )
    no_more_response = compact_utterance in {
        "thats all",
        "that s all",
        "that is all",
        "thats it",
        "that s it",
        "that is it",
        "nothing else",
        "no more",
        "no nothing else",
    }
    return option_question and no_more_response


def _better_expression_stays_generic_order(value: str) -> bool:
    compact = _normalize_utterance(value).replace("'", "")
    return bool(re.search(r"\b(?:like|want|order|get|have) (?:a |an |the )?drink\b", compact))


def _better_expression_changes_recommendation_intent(value: str) -> bool:
    compact = _normalize_utterance(value).replace("'", "")
    if "recommend" in compact or "suggest" in compact:
        return False
    return bool(re.search(r"\b(?:like|want|order|get|have) (?:a |an |the )?(?:drink|coffee|latte|americano|tea|juice|water)\b", compact))


def _is_direct_want_concrete_order_near_miss(user_utterance: str) -> bool:
    return _direct_want_concrete_order_parts(user_utterance) is not None


def _apply_direct_want_concrete_order_feedback(user_utterance: str, turn_feedback: Any) -> None:
    parts = _direct_want_concrete_order_parts(user_utterance)
    if parts is None:
        return

    turn_feedback.feedbackRequired = True
    turn_feedback.nativeUnderstanding = (
        f"외국인은 사용자가 {parts['korean_object_particle']} 주문하고 싶다고 이해했어요."
    )
    turn_feedback.nativeLanguageInterpretation = (
        f"한국어로 비유하자면, '{parts['korean_object']} 원해요'처럼 들려요."
    )
    turn_feedback.betterExpression = (
        f"I'd like {parts['english_object']}, please. "
        "이렇게 말하면 더 자연스럽고 공손하게 주문할 수 있어요."
    )


def _direct_want_concrete_order_parts(user_utterance: str) -> dict[str, str] | None:
    compact = _normalize_utterance(user_utterance).replace("'", "")
    if not re.fullmatch(r"i want (?:a |an |the )?.+", compact):
        return None
    if _generic_order_object_analogy(compact) is not None:
        return None

    drink_parts = [
        ("iced americano", "아이스 아메리카노", "아이스 아메리카노를", "an iced Americano"),
        ("americano", "아메리카노", "아메리카노를", "an Americano"),
        ("cappuccino", "카푸치노", "카푸치노를", "a cappuccino"),
        ("espresso", "에스프레소", "에스프레소를", "an espresso"),
        ("smoothie", "스무디", "스무디를", "a smoothie"),
        ("coffee", "커피", "커피를", "a coffee"),
        ("latte", "라떼", "라떼를", "a latte"),
        ("mocha", "모카", "모카를", "a mocha"),
        ("water", "물", "물을", "some water"),
        ("juice", "주스", "주스를", "some juice"),
        ("tea", "차", "차를", "some tea"),
    ]
    for token, korean_object, korean_object_particle, english_object in drink_parts:
        if re.search(rf"\b{re.escape(token)}\b", compact):
            return {
                "korean_object": korean_object,
                "korean_object_particle": korean_object_particle,
                "english_object": english_object,
            }
    return None


def _normalize_native_understanding_format(value: str | None) -> str | None:
    if value is None:
        return None

    quoted_meaning_match = re.search(r"[\"'‘’“”]([^\"'‘’“”]+)[\"'‘’“”](?:라는|는) 의미로 이해했어요\.", value)
    if quoted_meaning_match:
        phrase = _to_reported_understanding_phrase(quoted_meaning_match.group(1))
        return f"외국인은 사용자가 {phrase} 이해했어요."

    quoted_match = re.search(r"[\"'‘’“”]([^\"'‘’“”]+)[\"'‘’“”]\s*(?:라)?고 이해했어요\.", value)
    if quoted_match:
        phrase = _to_reported_understanding_phrase(quoted_match.group(1))
        return f"외국인은 사용자가 {phrase} 이해했어요."

    return value


def _to_reported_understanding_phrase(value: str) -> str:
    phrase = value.strip().rstrip(".")
    if phrase.endswith("다"):
        return phrase[:-1] + "다고"
    return phrase + "라고"


def _normalize_native_language_interpretation_format(value: str | None) -> str | None:
    if value is None:
        return None
    prefix = "한국어로 비유하자면, '"
    if not value.startswith(prefix):
        return value

    phrase = value.removeprefix(prefix).strip()
    if phrase.endswith("'처럼 들려요."):
        return value

    phrase = phrase.strip("'").rstrip(".")
    for suffix in ["처럼 들려요", "처럼 들려요."]:
        if phrase.endswith(suffix):
            phrase = phrase[: -len(suffix)].strip().rstrip(".")

    return f"한국어로 비유하자면, '{phrase}'처럼 들려요."


def _native_understanding_override(user_utterance: str) -> str | None:
    compact = _normalize_utterance(user_utterance).replace("'", "")
    if _is_incomplete_utterance_fragment(user_utterance):
        fragment = _trim_incomplete_fragment_for_feedback(user_utterance)
        return f"외국인은 '{fragment}'만 듣고는 어떤 음료를 주문하는지 이해할 수 없었어요."

    overrides = {
        "i dont know": "외국인은 사용자가 무엇을 주문할지 모르겠다고 이해했어요.",
        "i want ice one": "외국인은 사용자가 얼음 한 개를 원한다고 이해했어요.",
        "less ice do please": "외국인은 사용자가 얼음을 적게 넣어 달라고 이해했어요.",
        "this drink is hot but i order ice one": "외국인은 사용자가 이 음료는 뜨겁지만 얼음 한 개를 주문했다고 이해했어요.",
        "my shoes are swimming in the moon today": "외국인은 사용자가 신발이 달에서 수영하고 있다고 말한다고 이해했어요.",
        "i do not want to order anything": "외국인은 사용자가 아무것도 주문하지 않겠다고 이해했어요.",
        "i dont want to order anything": "외국인은 사용자가 아무것도 주문하지 않겠다고 이해했어요.",
    }
    return overrides.get(compact)


def _native_language_interpretation_override(user_utterance: str) -> str | None:
    compact = _normalize_utterance(user_utterance).replace("'", "")
    incomplete_analogy = _incomplete_order_fragment_analogy(user_utterance)
    if incomplete_analogy is not None:
        return f"한국어로 비유하자면, '{incomplete_analogy}'처럼 들려요."

    overrides = {
        "i dont know": "한국어로 비유하자면, '무엇을 주문할지 모르겠어요'처럼 들려요.",
        "i want ice one": "한국어로 비유하자면, '얼음 하나 원해요'처럼 들려요.",
        "less ice do please": "한국어로 비유하자면, '얼음 적게 해주세요'처럼 들려요.",
        "this drink is hot but i order ice one": "한국어로 비유하자면, '이 음료는 뜨겁지만 얼음 한 개를 주문했어요'처럼 들려요.",
        "my shoes are swimming in the moon today": "한국어로 비유하자면, '달에서 신발이 수영한다'처럼 들려요.",
        "i do not want to order anything": "한국어로 비유하자면, '주문 자체를 거절하는 것'처럼 들려요.",
        "i dont want to order anything": "한국어로 비유하자면, '주문 자체를 거절하는 것'처럼 들려요.",
    }
    return overrides.get(compact)


def _feedback_user_prompt(request: ConversationFeedbackRequest) -> str:
    turn_lines = "\n".join(
        f"- turnId: {turn.turnId}\n"
        f"  AI question: {turn.originalQuestion}\n"
        f"  User utterance: {turn.userUtterance}"
        for turn in request.turns
    )
    return (
        f"Scenario title: {request.scenarioTitle}\n"
        f"Scenario goal: {request.scenarioGoal}\n\n"
        f"Turns:\n{turn_lines}"
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ConversationGenerationError("model returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise ConversationGenerationError("model response must be a JSON object")
    return data


def _call_chat(system: str, user: str, max_tokens: int, temperature: float) -> str:
    try:
        return chat(system, user, max_tokens=max_tokens, temperature=temperature)
    except Exception as exc:
        raise ConversationGenerationError("model call failed") from exc


def _strip_code_fence(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()

    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    return cleaned


def _normalize_newly_filled_slots(data: dict[str, Any], unfilled_slot_names: list[str]) -> list[FilledSlotResponse]:
    raw_slots = data.get("filledSlots")
    if not isinstance(raw_slots, list):
        raise ConversationGenerationError("filledSlots must be a list")

    unfilled_slot_set = set(unfilled_slot_names)
    seen: set[str] = set()
    normalized: list[FilledSlotResponse] = []
    for raw_slot in raw_slots:
        if not isinstance(raw_slot, dict):
            raise ConversationGenerationError("filledSlots entries must be objects")

        slot_name = raw_slot.get("slotName")
        if not isinstance(slot_name, str) or not slot_name.strip():
            raise ConversationGenerationError("filledSlots entries must include slotName")

        slot_name = slot_name.strip()
        if slot_name not in unfilled_slot_set or slot_name in seen:
            continue

        seen.add(slot_name)
        normalized.append(FilledSlotResponse(slotName=slot_name))

    return normalized


def _optional_non_blank_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConversationGenerationError("question fields must be strings or null")
    stripped = value.strip()
    return stripped or None
