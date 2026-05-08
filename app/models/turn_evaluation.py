# 턴 평가 Pydantic 스키마 — 턴 단위 평가 요청/응답 데이터 구조 정의
from typing import Literal
from pydantic import BaseModel, Field


class FilledSlot(BaseModel):
    slotKey: str
    slotValue: str


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ScenarioSlot(BaseModel):
    slotKey: str
    description: str | None = None


class ScenarioPayload(BaseModel):
    scenarioId: str
    title: str | None = None
    situationDescription: str
    successGoal: str
    maxFollowUpCount: int = 5
    requiredSlots: list[ScenarioSlot] = Field(default_factory=list)


class CurrentQuestion(BaseModel):
    questionText: str
    ttsUrl: str | None = None


class TurnMeta(BaseModel):
    turnIndex: int
    inputType: Literal["AUDIO"]
    speechStartedAfterMs: int
    recordingDurationMs: int


class TurnEvaluationRequest(BaseModel):
    sessionId: str
    scenario: ScenarioPayload
    currentQuestion: CurrentQuestion
    currentFilledSlots: dict | list | None = None
    turn: TurnMeta
    conversationHistory: list[ConversationTurn] = Field(default_factory=list)


class TtsContent(BaseModel):
    questionText: str | None = None
    messageText: str | None = None
    ttsAudio: str  # base64


class TurnEvaluationResponse(BaseModel):
    transcript: str
    sttConfidence: float
    scenarioStatus: Literal["IN_PROGRESS", "SUCCESS", "FAILURE"]
    filledSlots: list[FilledSlot]  # 이번 턴에 새로 채워진 슬롯
    nextQuestion: TtsContent | None = None
    resultMessage: TtsContent | None = None
