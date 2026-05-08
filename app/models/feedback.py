# 피드백 Pydantic 스키마 — 세션 종료 후 피드백 응답 데이터 구조 정의
from pydantic import BaseModel
from app.models.conversation import Utterance


class FeedbackData(BaseModel):
    session_id: str  # uuid
    scenario_id: str  # ex: cafe_01
    cleared: bool  # 시나리오 클리어 여부
    total_comprehension: int  # 전체 발화 이해도 평균
    utterances: list[Utterance]  # 발화마다 피드백 반복
    fail_reason: str | None = None  # cleared=false일 때 실패 이유 (한글)


class FeedbackResponse(BaseModel):
    success: bool
    data: FeedbackData | None = None
    error: str | None = None
