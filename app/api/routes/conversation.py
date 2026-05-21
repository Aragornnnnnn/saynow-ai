# 2차 MVP 백엔드 연동용 대화 API 라우터를 제공한다.
import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.logger import get_logger
from app.models.conversation import (
    ConversationFeedbackRequest,
    ConversationFeedbackResponse,
    NextQuestionRequest,
    NextQuestionResponse,
)
from app.services.conversation_service import (
    ConversationGenerationError,
    generate_feedback,
    generate_feedback_stream_events,
    generate_next_question,
)


router = APIRouter(prefix="/api/v1/conversation", tags=["conversation"])
logger = get_logger("route.conversation")


@router.post(
    "/next-question",
    response_model=NextQuestionResponse,
    summary="꼬리 질문 생성",
)
async def next_question(request: NextQuestionRequest):
    logger.info(
        "POST /api/v1/conversation/next-question | scenario: %s | slots: %d",
        request.scenarioTitle,
        len(request.slots),
    )
    try:
        return generate_next_question(request)
    except ConversationGenerationError as exc:
        logger.error("꼬리 질문 생성 실패 | error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "code": "AI_GENERATION_FAILED",
                "message": "꼬리 질문 생성에 실패했습니다.",
            },
        )


@router.post(
    "/feedback",
    response_model=ConversationFeedbackResponse,
    summary="대화 피드백 생성",
)
async def feedback(request: ConversationFeedbackRequest):
    logger.info(
        "POST /api/v1/conversation/feedback | scenario: %s | turns: %d",
        request.scenarioTitle,
        len(request.turns),
    )
    try:
        return generate_feedback(request)
    except ConversationGenerationError as exc:
        logger.error("피드백 생성 실패 | error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "code": "AI_GENERATION_FAILED",
                "message": "피드백 생성에 실패했습니다.",
            },
        )


@router.post(
    "/feedback/stream",
    summary="대화 피드백 스트리밍 생성",
)
async def feedback_stream(request: ConversationFeedbackRequest):
    logger.info(
        "POST /api/v1/conversation/feedback/stream | scenario: %s | turns: %d",
        request.scenarioTitle,
        len(request.turns),
    )

    def event_generator():
        try:
            for event, data in generate_feedback_stream_events(request):
                yield _format_sse_event(event, data)
        except ConversationGenerationError as exc:
            logger.error("피드백 스트리밍 생성 실패 | error: %s", exc)
            yield _format_sse_event(
                "error",
                {
                    "code": "AI_GENERATION_FAILED",
                    "message": "피드백 생성에 실패했습니다.",
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
