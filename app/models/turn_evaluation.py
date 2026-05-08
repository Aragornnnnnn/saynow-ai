# 턴 평가 Pydantic 스키마 — 턴 단위 평가 요청/응답 데이터 구조 정의
from typing import Literal
from pydantic import BaseModel


class FilledSlot(BaseModel):
    slotKey: str
    slotValue: str


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


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
