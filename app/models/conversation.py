# 2차 MVP 대화 API 요청과 응답 데이터 구조를 정의한다.
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class EvidencePolicyMode(StrEnum):
    SEMANTIC_EVIDENCE = "semantic_evidence"
    EXPLICIT_PATTERN = "explicit_pattern"
    EXPLICIT_KEYWORD = "explicit_keyword"


class EvidenceGrounding(StrEnum):
    LATEST_USER_UTTERANCE = "latest_user_utterance"


class EvidencePolicy(BaseModel):
    mode: EvidencePolicyMode
    hints: list[str] = Field(default_factory=list)
    requiresEvidenceText: bool = True
    mustBeGroundedIn: EvidenceGrounding = EvidenceGrounding.LATEST_USER_UTTERANCE

    @field_validator("hints")
    @classmethod
    def hints_must_not_include_blank_values(cls, value: list[str]) -> list[str]:
        return [_validate_not_blank(hint) for hint in value]


class SlotStatusRequest(BaseModel):
    slotName: str
    description: str
    filled: bool
    evidencePolicy: EvidencePolicy | None = None

    @field_validator("slotName", "description")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
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
    REPEAT_REQUEST = "REPEAT_REQUEST"
    INVALID_RESPONSE = "INVALID_RESPONSE"


class SessionResult(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class NextQuestionRequest(BaseModel):
    originalQuestion: str
    originalQuestionTargetSlotName: str | None = None
    userUtterance: str
    scenarioTitle: str
    scenarioSituation: str
    aiRole: str
    scenarioGoal: str
    slots: list[SlotStatusRequest]

    @field_validator("originalQuestion", "userUtterance", "scenarioTitle", "scenarioSituation", "aiRole", "scenarioGoal")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        return _validate_not_blank(value)

    @field_validator("originalQuestionTargetSlotName")
    @classmethod
    def optional_target_slot_name_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_not_blank(value)


class NextQuestionResponse(BaseModel):
    nextQuestion: str | None
    translatedQuestion: str | None
    nextQuestionTargetSlotName: str | None = None
    filledSlots: list[FilledSlotResponse]
    turnClassification: NextQuestionTurnClassification

    @field_validator("nextQuestionTargetSlotName")
    @classmethod
    def optional_target_slot_name_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_not_blank(value)


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
    scenarioSituation: str
    aiRole: str
    scenarioGoal: str
    sessionResult: SessionResult
    slots: list[SlotStatusRequest]
    turns: list[FeedbackTurnRequest]

    @field_validator("scenarioTitle", "scenarioSituation", "aiRole", "scenarioGoal")
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
