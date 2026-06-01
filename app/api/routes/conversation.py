# 3차 MVP 백엔드 연동용 프리톡 대화 API 라우터를 제공한다.
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.logger import get_logger
from app.core.observability import capture_exception
from app.models.conversation import (
    GuideChatRequest,
    GuideChatResponse,
    NextQuestionRequest,
    NextQuestionResponse,
    SessionFeedbackRequest,
    SessionFeedbackResponse,
    TurnFeedbackCreationResponse,
    TurnFeedbackRequest,
)
from app.services.conversation_service import (
    ConversationGenerationError,
    TurnFeedbackNotReadyError,
    generate_guide_answer,
    generate_next_question,
    generate_session_feedback,
    generate_turn_feedback,
)


router = APIRouter(prefix="/api/v1/conversation", tags=["conversation"])
logger = get_logger("route.conversation")


@router.post(
    "/next-question",
    response_model=NextQuestionResponse,
    summary="다음 고정 질문 연결 문장 생성",
)
async def next_question(request: NextQuestionRequest):
    logger.info(
        "POST /api/v1/conversation/next-question | sessionId=%s scenarioId=%s submittedTurnId=%s",
        request.sessionId,
        request.scenario.scenarioId,
        request.submittedTurnId,
    )
    try:
        return generate_next_question(request)
    except ConversationGenerationError as exc:
        return _generation_error_response(exc, "다음 질문 생성에 실패했습니다.")


@router.post(
    "/turn-feedback",
    response_model=TurnFeedbackCreationResponse,
    summary="턴별 피드백 생성 및 AI 캐시 저장",
)
async def turn_feedback(request: TurnFeedbackRequest):
    logger.info(
        "POST /api/v1/conversation/turn-feedback | sessionId=%s turnId=%s sequence=%s",
        request.sessionId,
        request.turnId,
        request.sequence,
    )
    try:
        return generate_turn_feedback(request)
    except ConversationGenerationError as exc:
        return _generation_error_response(exc, "턴별 피드백 생성에 실패했습니다.")


@router.post(
    "/session-feedback",
    response_model=SessionFeedbackResponse,
    summary="세션 최종 피드백 생성",
)
async def session_feedback(request: SessionFeedbackRequest):
    logger.info(
        "POST /api/v1/conversation/session-feedback | sessionId=%s expectedTurnCount=%s",
        request.sessionId,
        len(request.expectedTurnIds),
    )
    try:
        return generate_session_feedback(request)
    except TurnFeedbackNotReadyError as exc:
        logger.info(
            "세션 피드백 생성 대기 | sessionId=%s missingTurnIds=%s",
            request.sessionId,
            exc.missing_turn_ids,
        )
        return JSONResponse(
            status_code=409,
            content={
                "code": "TURN_FEEDBACK_NOT_READY",
                "message": "턴별 피드백이 아직 준비되지 않았습니다.",
                "missingTurnIds": exc.missing_turn_ids,
            },
        )
    except ConversationGenerationError as exc:
        return _generation_error_response(exc, "세션 최종 피드백 생성에 실패했습니다.")


@router.post(
    "/guide",
    response_model=GuideChatResponse,
    summary="영어 학습 가이드 답변 생성",
)
async def guide(request: GuideChatRequest):
    logger.info(
        "POST /api/v1/conversation/guide | scenario: %s",
        request.scenarioTitle,
    )
    try:
        return generate_guide_answer(request)
    except ConversationGenerationError as exc:
        return _generation_error_response(exc, "가이드 답변 생성에 실패했습니다.")


def _generation_error_response(exc: ConversationGenerationError, message: str) -> JSONResponse:
    logger.exception("AI 생성 실패 | error: %s", exc)
    capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content={
            "code": "AI_GENERATION_FAILED",
            "message": message,
        },
    )
