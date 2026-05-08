# 피드백 라우터 — GET /feedback/{session_id} (완료된 세션의 피드백 데이터 반환)
from fastapi import APIRouter
from app.models.feedback import FeedbackResponse
from app.services.feedback_service import build_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.get("/{session_id}", response_model=FeedbackResponse, summary="세션 피드백 조회")
def get_feedback(session_id: str):
    """
    완료된 대화 세션의 상세 피드백을 조회합니다.

    - session_id: 세션 ID
    - 반환: 전체 이해도, 각 발화별 이해도·네이티브 인식·개선 표현
    """
    try:
        data = build_feedback(session_id)
        return FeedbackResponse(success=True, data=data)
    except ValueError as e:
        return FeedbackResponse(success=False, error=str(e))
