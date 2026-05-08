# 피드백 서비스 — 완료된 세션에서 발화 데이터를 집계해 피드백 응답을 만드는 로직
from app.models.conversation import Utterance
from app.models.feedback import FeedbackData
from app.services.conversation_service import get_session


def build_feedback(session_id: str) -> FeedbackData:
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found")
    if not session["done"]:
        raise ValueError("Session is not finished yet")

    utterances = [Utterance(**u) for u in session["utterances"]]
    scores = [u.comprehension_score for u in utterances]
    total = round(sum(scores) / len(scores)) if scores else 0

    return FeedbackData(
        session_id=session_id,
        scenario_id=session["scenario"]["id"],
        cleared=session["cleared"],
        total_comprehension=total,
        utterances=utterances,
    )
