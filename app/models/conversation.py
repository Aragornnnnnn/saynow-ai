# 3차 MVP 프리톡 대화 API 요청과 응답 데이터 구조를 정의한다.
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _optional_not_blank(value: str | None) -> str | None:
    if value is None:
        return None
    return _validate_not_blank(value)


class ScenarioContext(BaseModel):
    scenarioId: int = Field(gt=0)
    title: str
    briefing: str
    conversationGoal: str

    @field_validator("title", "briefing", "conversationGoal")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class CurrentTurnForNextQuestion(BaseModel):
    aiQuestion: str
    translatedQuestion: str
    userUtterance: str

    @field_validator("aiQuestion", "translatedQuestion", "userUtterance")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class NextFixedQuestion(BaseModel):
    questionId: int = Field(gt=0)
    sequence: int = Field(gt=0)
    questionEn: str
    questionKo: str

    @field_validator("questionEn", "questionKo")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class NextQuestionRequest(BaseModel):
    sessionId: int = Field(gt=0)
    submittedTurnId: int = Field(gt=0)
    submittedSequence: int = Field(gt=0)
    scenario: ScenarioContext
    currentTurn: CurrentTurnForNextQuestion
    nextQuestion: NextFixedQuestion


class NextQuestionResponse(BaseModel):
    aiQuestion: str
    translatedQuestion: str

    @field_validator("aiQuestion", "translatedQuestion")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class TurnForFeedback(BaseModel):
    aiQuestion: str
    translatedQuestion: str
    userUtterance: str

    @field_validator("aiQuestion", "translatedQuestion", "userUtterance")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class TurnFeedbackRequest(BaseModel):
    sessionId: int = Field(gt=0)
    turnId: int = Field(gt=0)
    sequence: int = Field(gt=0)
    scenario: ScenarioContext
    turn: TurnForFeedback


class TurnFeedbackStatus(StrEnum):
    PREPARING = "PREPARING"
    READY = "READY"
    FAILED = "FAILED"


class TurnFeedbackCreationResponse(BaseModel):
    sessionId: int
    turnId: int
    feedbackStatus: TurnFeedbackStatus


class FeedbackType(StrEnum):
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    GOOD = "GOOD"


class TurnFeedbackData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turnId: int = Field(gt=0)
    feedbackType: FeedbackType
    koreanAnalogy: str
    positiveFeedback: str | None = None
    feedbackDetail: str
    benchmarkMessage: str | None = None

    @field_validator(
        "koreanAnalogy",
        "positiveFeedback",
        "feedbackDetail",
        "benchmarkMessage",
    )
    @classmethod
    def optional_text_fields_must_not_be_blank(cls, value: str | None) -> str | None:
        return _optional_not_blank(value)

    @model_validator(mode="after")
    def feedback_fields_must_match_type(self):
        if self.feedbackType == FeedbackType.NEEDS_IMPROVEMENT:
            if self.positiveFeedback is None or not self.positiveFeedback.strip():
                raise ValueError("positiveFeedback is required for NEEDS_IMPROVEMENT feedback")
            if self.benchmarkMessage is not None:
                raise ValueError("benchmarkMessage must be null for NEEDS_IMPROVEMENT feedback")
            return self

        if self.positiveFeedback is not None:
            raise ValueError("positiveFeedback must be null for GOOD feedback")
        return self


class SessionFeedbackRequest(BaseModel):
    sessionId: int = Field(gt=0)
    scenario: ScenarioContext
    expectedTurnIds: list[int]

    @field_validator("expectedTurnIds")
    @classmethod
    def expected_turn_ids_must_not_be_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("expectedTurnIds must not be empty")
        if any(turn_id <= 0 for turn_id in value):
            raise ValueError("expectedTurnIds must contain positive ids")
        if len(value) != len(set(value)):
            raise ValueError("expectedTurnIds must not contain duplicates")
        return value


class SessionFeedbackHighlightResponse(BaseModel):
    sessionId: int = Field(gt=0)
    highlightMessage: str

    @field_validator("highlightMessage")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class NativeScoreBreakdown(BaseModel):
    attemptedWordScore: int = Field(ge=0, le=100)
    sentenceComplexityScore: int = Field(ge=0, le=100)
    comprehensibilityScore: int = Field(ge=0, le=100)


class SessionFeedbackResponse(BaseModel):
    sessionId: int = Field(gt=0)
    nativeScore: int = Field(ge=0, le=100)
    highlightMessage: str
    turnFeedbacks: list[TurnFeedbackData]

    @field_validator("highlightMessage")
    @classmethod
    def highlight_message_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class GuideChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    scenarioTitle: str
    scenarioSituation: str
    aiRole: str
    scenarioGoal: str

    @field_validator("question", "scenarioTitle", "scenarioSituation", "aiRole", "scenarioGoal")
    @classmethod
    def required_text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class GuideChatResponse(BaseModel):
    answer: str

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)
