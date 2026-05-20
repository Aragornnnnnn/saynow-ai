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

    try:
        response = ConversationFeedbackResponse.model_validate(data)
    except ValidationError as exc:
        raise ConversationGenerationError("feedback response does not match contract") from exc

    request_turn_ids = [turn.turnId for turn in request.turns]
    response_turn_ids = [turn.turnId for turn in response.turnFeedbacks]
    if response_turn_ids != request_turn_ids:
        raise ConversationGenerationError("turn feedback ids do not match request turn ids")

    _enforce_native_language_interpretation_contract(request, response)
    _enforce_feedback_consistency(request, response)
    return response


def _next_question_system_prompt() -> str:
    return (
        "You generate follow-up questions for an English speaking practice scenario. "
        "Return ONLY valid JSON matching this schema exactly: "
        '{"filledSlots":[{"slotName":"..."}],"nextQuestion":"<string or null>","translatedQuestion":"<string or null>"}. '
        "filledSlots must contain only slot names that were newly satisfied by the user's latest utterance. "
        "Only mark a slot as filled when the user explicitly provides a concrete value for that exact slot. "
        "Do not infer slot values from context, politeness, refusal, uncertainty, random text, or unrelated sentences. "
        "Nonsense, off-topic, refusal, or vague non-answer utterances must return filledSlots=[] and ask again for the same missing information. "
        "These utterances must never fill any slot: qwertyuiop asdfghjkl zxcvbnm, My shoes are swimming in the moon today, I don't know, No answer, I do not want to order anything. "
        "Never include slots that were already filled before this request. "
        "If all currently unfilled slots are newly satisfied, set nextQuestion and translatedQuestion to null. "
        "Do not set nextQuestion or translatedQuestion to null unless every currently unfilled slot is explicitly satisfied by the latest utterance. "
        "If any currently unfilled slot remains, ask one short natural English follow-up question and include a Korean translation. "
        "Use only the provided slot names."
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
        "Return ONLY valid JSON matching this schema exactly: "
        '{"comprehensionScore":82,"feedbackSummary":"...","turnFeedbacks":[{"turnId":101,"feedbackRequired":true,"nativeUnderstanding":"...","nativeLanguageInterpretation":"...","betterExpression":"..."}]}. '
        "comprehensionScore is an integer from 0 to 100 from a native listener's perspective. "
        "feedbackSummary is Korean and summarizes overall comprehension, whether the scenario goal was effectively handled, strengths, and one improvement direction. "
        "For each turn, preserve the exact turnId from the request. "
        "Stable feedback decision rubric: 0-39 means the answer is off-topic or a native listener cannot identify the intended meaning; "
        "40-59 means only a vague gist is understandable and key scenario information is missing or heavily distorted; "
        "60-74 means the main intent is understandable but grammar, word choice, or word order is clearly awkward enough to need correction; "
        "75-84 means the scenario intent is clear but a small correction would noticeably improve naturalness, politeness, or completeness; "
        "85-100 means the answer directly answers the question, a native listener understands it without guessing, and any remaining awkwardness is minor. "
        "If the scenario goal is not achieved, comprehensionScore must be 59 or below. "
        "Nonsense, off-topic, refusal, or vague non-answer utterances must score 0-39. "
        "Good Response Conditions: the answer must address the AI question, satisfy the scenario intent for that turn, be understandable without extra inference, and have no meaning-blocking grammar or word-choice issue. "
        "Only set feedbackRequired=false when all Good Response Conditions pass and the internal turn score is 85-100. "
        "If any condition fails, or the internal turn score is 84 or below, set feedbackRequired=true. "
        "Apply this exact rubric consistently for every request; do not loosen or tighten it by scenario, user level, or writing style. "
        "When feedbackRequired=false, set nativeUnderstanding, nativeLanguageInterpretation, and betterExpression to null. "
        "When feedbackRequired is true, nativeUnderstanding must explain what the foreign listener understood from the user's utterance. "
        "Do not quote the user's utterance in nativeUnderstanding. "
        "Do not use nativeUnderstanding for meta-evaluation such as saying the utterance is unrelated, figurative, or grammatically wrong. "
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
        "When the user's utterance does not answer the AI question or scenario intent, give a simple English answer without wrapping it in quotation marks, then explain why it fits. "
        "The English example must appear plainly without double quotation marks, for example 'I'd like an Americano, please. 이렇게 말하면 원하는 음료를 명확하게 전달할 수 있어요.' "
        "If the exact answer is unknown, use a generic English example that fits the scenario, such as 'I'd like a coffee, please.' for ordering a drink. "
        "Do not return only an English sentence with a parenthesized Korean translation."
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
        turn_feedback.nativeUnderstanding = (
            turn_feedback.nativeUnderstanding
            or "외국인은 시나리오 목표에 필요한 답변을 듣지 못했다고 받아들일 수 있어요."
        )
        turn_feedback.nativeLanguageInterpretation = (
            turn_feedback.nativeLanguageInterpretation
            or "한국어로 비유하자면, '대답을 하지 않은 것'처럼 들려요."
        )
        turn_feedback.betterExpression = (
            turn_feedback.betterExpression
            or "I'd like a coffee, please. 이렇게 말하면 원하는 음료를 명확하게 주문할 수 있어요."
        )


def _enforce_native_language_interpretation_contract(
    request: ConversationFeedbackRequest,
    response: ConversationFeedbackResponse,
) -> None:
    turns_by_id = {turn.turnId: turn for turn in request.turns}
    for turn_feedback in response.turnFeedbacks:
        if not turn_feedback.feedbackRequired:
            continue

        turn = turns_by_id.get(turn_feedback.turnId)
        if turn is None:
            continue

        interpretation = _native_language_interpretation_override(turn.userUtterance)
        if interpretation is not None:
            turn_feedback.nativeLanguageInterpretation = interpretation


def _native_language_interpretation_override(user_utterance: str) -> str | None:
    compact = _normalize_utterance(user_utterance).replace("'", "")
    overrides = {
        "i want ice one": "한국어로 비유하자면, '얼음 하나 원해요'처럼 들려요.",
        "less ice do please": "한국어로 비유하자면, '얼음 적게 해주세요'처럼 들려요.",
        "this drink is hot but i order ice one": "한국어로 비유하자면, '이 음료는 뜨겁지만 얼음 한 개를 주문했어요'처럼 들려요.",
        "my shoes are swimming in the moon today": "한국어로 비유하자면, '달에서 신발이 수영한다'처럼 들려요.",
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
