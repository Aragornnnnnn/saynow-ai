# 세션 피드백 라우터 — POST /api/v1/session-feedbacks
from fastapi import APIRouter
from app.core.logger import get_logger
from app.models.session_feedback import SessionFeedbackRequest, SessionFeedbackResponse
from app.services.session_feedback_service import build_feedback

router = APIRouter()
logger = get_logger("route.session_feedback")


@router.post(
    "/api/v1/session-feedbacks",
    response_model=SessionFeedbackResponse,
    summary="세션 최종 피드백 생성",
    description="""
롤플레이 세션이 종료된 후, 전체 대화 기록을 분석하여 종합 피드백을 생성합니다.

**처리 흐름**
1. 세션의 모든 턴(질문 + 유저 발화)을 LLM에 전달
2. 각 턴별 이해도 점수(understoodScore), 원어민에게 들린 내용(heardAs), 더 나은 표현(betterExpression) 분석
3. 전체 세션 총점(totalUnderstoodScore) 및 종합 코멘트(summary) 생성

**scenarioResult 값**
- `SUCCESS`: 필수 슬롯을 모두 충족하여 성공한 세션
- `FAILURE`: 질문 횟수 초과로 실패한 세션

**응답 필드**
- `totalUnderstoodScore`: 0~100, 세션 전체 평균 이해도 점수
- `summary`: 한글로 작성된 전체 피드백 코멘트
- `turns[].scoreDelta`: betterExpression을 사용했을 때 예상 점수 상승폭
- `turns[].improvedUnderstoodScore`: 개선 표현 적용 시 예상 점수
""",
)
async def session_feedback(request: SessionFeedbackRequest) -> SessionFeedbackResponse:
    logger.info("POST /api/v1/session-feedbacks (최종 피드백 생성 api) | session_id: %s | scenario: %s | turns: %d", request.sessionId, request.scenario.scenarioId, len(request.turns))
    try:
        return build_feedback(request)
    except Exception as e:
        logger.error("[500] 세션 피드백 생성 실패 — LLM 분석 또는 응답 JSON 파싱 단계에서 문제 발생 가능 | session_id: %s | error: %s", request.sessionId, e)
        raise
