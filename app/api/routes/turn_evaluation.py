# 턴 평가 라우터 — POST /api/v1/turn-evaluations
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from app.models.turn_evaluation import (
    FilledSlot,
    TurnEvaluationRequest,
    TurnEvaluationResponse,
)
from app.services.turn_evaluation_service import evaluate_turn

router = APIRouter()


@router.post("/api/v1/turn-evaluations", response_model=TurnEvaluationResponse)
async def turn_evaluation(
    audio: UploadFile = File(...),
    payload: str = Form(...),
):
    try:
        request = TurnEvaluationRequest.model_validate_json(payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid payload JSON: {e}")

    audio_bytes = await audio.read()
    filename = audio.filename or "audio.webm"
    slots: list[FilledSlot] = _parse_filled_slots(request.currentFilledSlots)
    history = [turn.model_dump() for turn in request.conversationHistory]
    required_keys = [slot.slotKey for slot in request.scenario.requiredSlots]

    try:
        return evaluate_turn(
            audio_bytes=audio_bytes,
            filename=filename,
            scenario_id=request.scenario.scenarioId,
            scenario_situation=request.scenario.situationDescription,
            scenario_goal=request.scenario.successGoal,
            required_keys=required_keys,
            max_follow_up_count=request.scenario.maxFollowUpCount,
            current_question=request.currentQuestion.questionText,
            filled_slots=slots,
            conversation_history=history,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _parse_filled_slots(raw: dict | list | None) -> list[FilledSlot]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [FilledSlot(**slot) for slot in raw if isinstance(slot, dict)]
    if isinstance(raw, dict):
        slots: list[FilledSlot] = []
        for slot_key, slot_value in raw.items():
            if isinstance(slot_value, dict) and "slotValue" in slot_value:
                slots.append(FilledSlot(slotKey=slot_key, slotValue=str(slot_value["slotValue"])))
            else:
                slots.append(FilledSlot(slotKey=slot_key, slotValue=str(slot_value)))
        return slots
    return []
