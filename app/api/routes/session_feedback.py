# 세션 피드백 라우터 — POST /api/v1/session-feedbacks
from fastapi import APIRouter
from app.models.session_feedback import SessionFeedbackRequest, SessionFeedbackResponse
from app.services.session_feedback_service import build_feedback

router = APIRouter()


@router.post("/api/v1/session-feedbacks", response_model=SessionFeedbackResponse)
async def session_feedback(request: SessionFeedbackRequest) -> SessionFeedbackResponse:
    return build_feedback(request)
