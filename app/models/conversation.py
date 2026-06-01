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
    turnId: int = Field(gt=0)
    feedbackType: FeedbackType
    koreanAnalogy: str
    correctionPoint: str | None = None
    correctionReason: str | None = None
    plusOneExpression: str | None = None
    praiseSummary: str | None = None
    praiseReason: str | None = None

    @field_validator(
        "koreanAnalogy",
        "correctionPoint",
        "correctionReason",
        "plusOneExpression",
        "praiseSummary",
        "praiseReason",
    )
    @classmethod
    def optional_text_fields_must_not_be_blank(cls, value: str | None) -> str | None:
        return _optional_not_blank(value)

    @model_validator(mode="after")
    def feedback_fields_must_match_type(self):
        if self.feedbackType == FeedbackType.NEEDS_IMPROVEMENT:
            required_values = [
                self.correctionPoint,
                self.correctionReason,
                self.plusOneExpression,
            ]
            if any(value is None or not value.strip() for value in required_values):
                raise ValueError("correction fields are required for NEEDS_IMPROVEMENT feedback")
            return self

        required_values = [self.praiseSummary, self.praiseReason]
        if any(value is None or not value.strip() for value in required_values):
            raise ValueError("praise fields are required for GOOD feedback")
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


class SessionFeedbackSummaryResponse(BaseModel):
    sessionId: int = Field(gt=0)
    nativeScore: int = Field(ge=0, le=100)
    nativeLevelLabel: str
    summary: str

    @field_validator("nativeLevelLabel", "summary")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class SessionFeedbackResponse(SessionFeedbackSummaryResponse):
    turnFeedbacks: list[TurnFeedbackData]


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
