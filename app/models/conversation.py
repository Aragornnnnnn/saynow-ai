# 2차 MVP 대화 API 요청과 응답 데이터 구조를 정의한다.
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


def _validate_not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class SlotStatusRequest(BaseModel):
    slotName: str
    filled: bool

    @field_validator("slotName")
    @classmethod
    def slot_name_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class FilledSlotResponse(BaseModel):
    slotName: str

    @field_validator("slotName")
    @classmethod
    def slot_name_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class NextQuestionTurnClassification(StrEnum):
    ANSWER = "ANSWER"
    ASSISTANCE_REQUEST = "ASSISTANCE_REQUEST"
    INVALID_RESPONSE = "INVALID_RESPONSE"


class SessionResult(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class NextQuestionRequest(BaseModel):
    originalQuestion: str
    userUtterance: str
    scenarioTitle: str
    scenarioGoal: str
    slots: list[SlotStatusRequest]

    @field_validator("originalQuestion", "userUtterance", "scenarioTitle", "scenarioGoal")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class NextQuestionResponse(BaseModel):
    nextQuestion: str | None
    translatedQuestion: str | None
    filledSlots: list[FilledSlotResponse]
    turnClassification: NextQuestionTurnClassification


class FeedbackTurnRequest(BaseModel):
    turnId: int
    originalQuestion: str
    userUtterance: str

    @field_validator("originalQuestion", "userUtterance")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class ConversationFeedbackRequest(BaseModel):
    scenarioTitle: str
    scenarioGoal: str
    sessionResult: SessionResult
    turns: list[FeedbackTurnRequest]

    @field_validator("scenarioTitle", "scenarioGoal")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)

    @field_validator("turns")
    @classmethod
    def turns_must_not_be_empty(cls, value: list[FeedbackTurnRequest]) -> list[FeedbackTurnRequest]:
        if not value:
            raise ValueError("turns must not be empty")
        return value


class TurnFeedbackResponse(BaseModel):
    turnId: int
    feedbackRequired: bool
    nativeUnderstanding: str | None = None
    nativeLanguageInterpretation: str | None = None
    betterExpression: str | None = None

    @model_validator(mode="after")
    def required_feedback_fields_must_exist_when_feedback_required(self):
        if not self.feedbackRequired:
            return self

        required_values = [
            self.nativeUnderstanding,
            self.nativeLanguageInterpretation,
            self.betterExpression,
        ]
        if any(value is None or not value.strip() for value in required_values):
            raise ValueError("feedback fields must exist when feedbackRequired is true")
        return self


class ConversationFeedbackSummaryResponse(BaseModel):
    comprehensionScore: int = Field(ge=0, le=100)
    feedbackSummary: str

    @field_validator("feedbackSummary")
    @classmethod
    def feedback_summary_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class ConversationFeedbackResponse(BaseModel):
    comprehensionScore: int = Field(ge=0, le=100)
    feedbackSummary: str
    turnFeedbacks: list[TurnFeedbackResponse]

    @field_validator("feedbackSummary")
    @classmethod
    def feedback_summary_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)
