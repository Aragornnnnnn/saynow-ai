# 턴 평가 라우터 — POST /api/v1/turn-evaluations
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from app.models.turn_evaluation import (
    FilledSlot,
    TurnEvaluationRequest,
    TurnEvaluationResponse,
)
from app.services.turn_evaluation_service import evaluate_turn

router = APIRouter()


@router.post(
    "/api/v1/turn-evaluations",
    response_model=TurnEvaluationResponse,
    summary="턴 단위 발화 평가",
    description="""
유저의 오디오 발화를 STT로 변환한 뒤, 시나리오 컨텍스트에 맞게 이해도를 평가하고 다음 질문을 생성합니다.

**Request DTO 형식**: `multipart/form-data`
- `audio`: 녹음된 오디오 파일 (webm, mp4 등)
- `payload`: JSON 문자열 (TurnEvaluationRequest 모델 -> sessionId, scenario, currentQuestion, currentFilledSlots, turn, conversationHistory 포함)

**처리 흐름**
1. 음성 녹음본 → Whisper STT → transcript 추출
2. 시나리오 슬롯 기준으로 이해도 분석 (LLM)
3. 미충족 슬롯이 있으면 꼬리질문 생성 → TTS 변환
4. 모든 슬롯 충족 또는 maxFollowUpCount 소진 시 종료

**response DTO**
- `transcript`: STT로 변환된 유저 발화 텍스트
- `sttConfidence`: 음성 인식 정확도 (0~ 1.0 사이) ex) sttConfidence = 0.58 → "이 오디오에서 내가 들은 게 맞는지 58% 확신"
- `scenarioStatus`: 대화 상태 (IN_PROGRESS, SUCCESS, FAILURE)
- `filledSlots`: 현재까지 채워진 슬롯 목록
- `nextQuestion`: 다음 질문 (scenarioStatus=IN_PROGRESS인 경우)
- `resultMessage`: 시나리오가 끝났을 때 (SUCCESS or FAILURE) nextQuestion 대신 나오는 마무리 메세지.

**scenarioStatus 값**
- `IN_PROGRESS`: 대화 진행 중
- `SUCCESS`: 필수 슬롯 모두 충족
- `FAILURE`: 질문 횟수 초과로 실패
""",
)
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
