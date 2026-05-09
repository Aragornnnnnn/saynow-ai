# 세션 피드백 Pydantic 스키마 — 세션 종료 후 최종 피드백 요청/응답 데이터 구조 정의
from typing import Literal
from pydantic import BaseModel


class ScenarioPayload(BaseModel):
    scenarioId: str
    title: str
    situationDescription: str
    successGoal: str


class FilledSlot(BaseModel):
    slotKey: str
    slotValue: str


class FeedbackTurn(BaseModel):
    turnId: int
    turnIndex: int
    questionText: str
    userTranscript: str
    speechStartedAfterMs: int | None = None
    recordingDurationMs: int | None = None


class SessionFeedbackRequest(BaseModel):
    sessionId: str
    scenario: ScenarioPayload
    scenarioResult: Literal["SUCCESS", "FAILURE"]
    filledSlots: list[FilledSlot]
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
