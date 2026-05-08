# 세션 피드백 Pydantic 스키마 — 세션 종료 후 최종 피드백 요청/응답 데이터 구조 정의
from pydantic import BaseModel


class FeedbackTurn(BaseModel):
    transcript: str
    question: str
    responseTimeSec: float


class SessionFeedbackRequest(BaseModel):
    scenarioId: str
    scenarioGoal: str
    turns: list[FeedbackTurn]


class TurnFeedback(BaseModel):
    understoodScore: int
    heardAs: str
    betterExpression: str
    scoreDelta: int
    improvedUnderstoodScore: int
    reason: str  # 한글


class SessionFeedbackResponse(BaseModel):
    totalUnderstoodScore: int
    summary: str  # 한글
    turns: list[TurnFeedback]
