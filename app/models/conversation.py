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


def _strip_korean_analogy_framing(value: str) -> str:
    stripped = value.strip()
    framing_prefixes = (
        "한국어로 비유하자면",
        "한국어로 비유하면",
        "한국어로 치면",
    )
    for prefix in framing_prefixes:
        if stripped.startswith(prefix):
            return stripped[len(prefix):].lstrip(" ,，:：")
    return stripped


class ServiceAudience(StrEnum):
    KOREAN_LEARNER = "KOREAN_LEARNER"
    AMERICAN_LEARNER = "AMERICAN_LEARNER"


class ScenarioContext(BaseModel):
    scenarioId: int = Field(gt=0)
    title: str
    briefing: str
    conversationGoal: str
    counterpartRole: str
    serviceAudience: ServiceAudience = ServiceAudience.KOREAN_LEARNER

    @field_validator("title", "briefing", "conversationGoal", "counterpartRole")
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


class InnerThoughtType(StrEnum):
    GOOD = "GOOD"
    NORMAL = "NORMAL"
    BAD = "BAD"


class NextQuestionResponse(BaseModel):
    aiQuestion: str
    translatedQuestion: str
    innerThought: str
    innerThoughtType: InnerThoughtType

    @field_validator("aiQuestion", "translatedQuestion", "innerThought", "innerThoughtType")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)


class ClosingReason(StrEnum):
    GOAL_COMPLETED = "GOAL_COMPLETED"
    MAX_TURNS_REACHED = "MAX_TURNS_REACHED"
    USER_ENDED = "USER_ENDED"
    TIME_LIMIT_REACHED = "TIME_LIMIT_REACHED"


class GoalCompletionStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    PARTIAL = "PARTIAL"
    COMPLETED = "COMPLETED"


class ClosingMessageRequest(BaseModel):
    sessionId: int = Field(gt=0)
    submittedTurnId: int = Field(gt=0)
    submittedSequence: int = Field(gt=0)
    scenario: ScenarioContext
    currentTurn: CurrentTurnForNextQuestion
    closingReason: ClosingReason
    goalCompletionStatus: GoalCompletionStatus


class ClosingMessageResponse(BaseModel):
    aiMessage: str
    translatedMessage: str
    innerThought: str
    innerThoughtType: InnerThoughtType

    @field_validator("aiMessage", "translatedMessage", "innerThought", "innerThoughtType")
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
    feedbackDetail: str | None = None
    correctionExpression: str | None = None
    correctionReason: str | None = None
    benchmarkMessage: str | None = None

    @field_validator("koreanAnalogy")
    @classmethod
    def korean_analogy_must_not_be_blank_or_framed(cls, value: str) -> str:
        return _validate_not_blank(_strip_korean_analogy_framing(value))

    @field_validator(
        "positiveFeedback",
        "feedbackDetail",
        "correctionExpression",
        "correctionReason",
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
            if self.feedbackDetail is not None:
                raise ValueError("feedbackDetail must be null for NEEDS_IMPROVEMENT feedback")
            if self.correctionExpression is None or not self.correctionExpression.strip():
                raise ValueError("correctionExpression is required for NEEDS_IMPROVEMENT feedback")
            if self.correctionReason is None or not self.correctionReason.strip():
                raise ValueError("correctionReason is required for NEEDS_IMPROVEMENT feedback")
            if self.benchmarkMessage is not None:
                raise ValueError("benchmarkMessage must be null for NEEDS_IMPROVEMENT feedback")
            return self

        if self.positiveFeedback is not None:
            raise ValueError("positiveFeedback must be null for GOOD feedback")
        if self.feedbackDetail is None or not self.feedbackDetail.strip():
            raise ValueError("feedbackDetail is required for GOOD feedback")
        if self.correctionExpression is not None:
            raise ValueError("correctionExpression must be null for GOOD feedback")
        if self.correctionReason is not None:
            raise ValueError("correctionReason must be null for GOOD feedback")
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

    serviceAudience: ServiceAudience = ServiceAudience.KOREAN_LEARNER
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
