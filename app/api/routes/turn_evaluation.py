# 턴 평가 라우터 — POST /api/v1/turn-evaluations
import json
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from app.models.turn_evaluation import FilledSlot, ConversationTurn, TurnEvaluationResponse
from app.services.turn_evaluation_service import evaluate_turn

router = APIRouter()


@router.post("/api/v1/turn-evaluations", response_model=TurnEvaluationResponse)
async def turn_evaluation(
    audio: UploadFile = File(...),
    scenarioId: str = Form(...),
    scenarioSituation: str = Form(...),
    scenarioGoal: str = Form(...),
    currentQuestion: str = Form(...),
    filledSlots: str = Form(default="[]"),
    conversationHistory: str = Form(default="[]"),
):
    try:
        slots: list[FilledSlot] = [FilledSlot(**s) for s in json.loads(filledSlots)]
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid filledSlots JSON: {e}")

    try:
        history: list[dict] = json.loads(conversationHistory)
        history = [ConversationTurn(**h).model_dump() for h in history]
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid conversationHistory JSON: {e}")

    audio_bytes = await audio.read()
    filename = audio.filename or "audio.webm"

    return evaluate_turn(
        audio_bytes=audio_bytes,
        filename=filename,
        scenario_id=scenarioId,
        scenario_situation=scenarioSituation,
        scenario_goal=scenarioGoal,
        current_question=currentQuestion,
        filled_slots=slots,
        conversation_history=history,
    )
