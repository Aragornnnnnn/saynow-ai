# 2차 MVP 대화 API의 LLM 호출과 응답 정규화를 담당한다.
import json
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

    raw = _call_chat(
        _next_question_system_prompt(),
        _next_question_user_prompt(request, unfilled_slot_names),
        max_tokens=512,
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

    return response


def _next_question_system_prompt() -> str:
    return (
        "You generate follow-up questions for an English speaking practice scenario. "
        "Return ONLY valid JSON matching this schema exactly: "
        '{"filledSlots":[{"slotName":"..."}],"nextQuestion":"<string or null>","translatedQuestion":"<string or null>"}. '
        "filledSlots must contain only slot names that were newly satisfied by the user's latest utterance. "
        "Never include slots that were already filled before this request. "
        "If all currently unfilled slots are newly satisfied, set nextQuestion and translatedQuestion to null. "
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
        "feedbackRequired is false when the user's response is already good enough; in that case set nativeUnderstanding, nativeLanguageInterpretation, and betterExpression to null. "
        "When feedbackRequired is true, nativeUnderstanding explains in Korean what a native listener understood. "
        "nativeLanguageInterpretation explains in Korean how the awkward English would sound if translated into the user's native language. "
        "betterExpression must be only one level more natural than the user's utterance, not an advanced rewrite."
    )


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


def _call_chat(system: str, user: str, max_tokens: int) -> str:
    try:
        return chat(system, user, max_tokens=max_tokens)
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
