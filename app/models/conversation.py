# 대화 Pydantic 스키마 — 대화 시작/진행 요청·응답 및 발화 데이터 정의
from pydantic import BaseModel


class StartRequest(BaseModel):
    scenario_id: str


class StartResponseData(BaseModel):
    session_id: str
    question: str
    audio_base64: str


class StartResponse(BaseModel):
    success: bool
    data: StartResponseData | None = None
    error: str | None = None


class NextRequest(BaseModel):
    session_id: str
    stt_text: str
    response_time_sec: float


class NextResponseData(BaseModel):
    done: bool
    cleared: bool | None = None
    next_question: str | None = None
    closing_message: str | None = None
    audio_base64: str | None = None
    utterance: dict | None = None


class NextResponse(BaseModel):
    success: bool
    data: NextResponseData | None = None
    error: str | None = None


class Utterance(BaseModel):
    question: str
    text: str
    response_time_sec: float
    comprehension_score: int
    native_perception: str
    better_expression: str
