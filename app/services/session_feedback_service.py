# 세션 피드백 서비스 — 전체 대화 턴을 분석해 최종 피드백 응답 생성
import json
from app.core.llm import chat
from app.models.session_feedback import (
    SessionFeedbackRequest,
    SessionFeedbackResponse,
    TurnFeedback,
)


def build_feedback(request: SessionFeedbackRequest) -> SessionFeedbackResponse:
    turn_feedbacks: list[TurnFeedback] = []
    prev_score = 0
    scenario_goal = request.scenario.successGoal

    for i, turn in enumerate(request.turns):
        analysis = _analyze_utterance(turn.userTranscript, request, turn)
        score = analysis["comprehension_score"]
        score_delta = score - prev_score if i > 0 else 0
        improved_score = _estimate_improved_score(
            analysis["better_expression"], scenario_goal, turn.questionText
        )
        reason = _generate_turn_reason(
            turn.userTranscript, score, analysis["better_expression"], scenario_goal
        )
        turn_feedbacks.append(
            TurnFeedback(
                understoodScore=score,
                heardAs=analysis["native_perception"],
                betterExpression=analysis["better_expression"],
                scoreDelta=score_delta,
                improvedUnderstoodScore=improved_score,
                reason=reason,
            )
        )
        prev_score = score

    total = round(sum(t.understoodScore for t in turn_feedbacks) / len(turn_feedbacks)) if turn_feedbacks else 0
    summary = _generate_summary(request)

    return SessionFeedbackResponse(
        totalUnderstoodScore=total,
        summary=summary,
        turns=turn_feedbacks,
    )


# 대화 분석 — LLM에게 STT된 유저 발화 (userTranscript)에 대한 이해도 점수, 원어민 인식 내용, 개선 표현 요청
def _analyze_utterance(transcript: str, request: SessionFeedbackRequest, turn) -> dict:
    system = (
        "You are an English language expert evaluating a non-native speaker's utterance. "
        "Respond ONLY with valid JSON matching this schema exactly:\n"
        '{"comprehension_score": <0-100 int>, "native_perception": "<string>", "better_expression": "<string>"}\n'
        "comprehension_score: rate the utterance on grammar correctness, naturalness, and fluency as a native American English speaker would judge it. "
        "Deduct points for: unnatural phrasing, missing articles, awkward word order, overly literal or robotic expressions. "
        "Do NOT give 100 unless the utterance is completely natural and idiomatic. "
        "A grammatically understandable but unnatural reply should score 60-80.\n"
        "native_perception: a short phrase describing what the native speaker actually heard/understood from the reply.\n"
        "better_expression: one improved alternative — more natural, idiomatic, and fluent for a native speaker."
    )
    user = (
        f"Question asked: {turn.questionText}\n"
        f"User's reply: {transcript}"
    )
    raw = chat(system, user, max_tokens=256)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"comprehension_score": 50, "native_perception": raw[:100], "better_expression": ""}


def _estimate_improved_score(better_expression: str, scenario_goal: str, question: str) -> int:
    system = (
        "You are an English language expert. "
        "If the user had said the following improved expression, what comprehension_score (0-100) "
        "would a native American English speaker give? "
        'Respond ONLY with valid JSON: {"score": <int>}'
    )
    user = (
        f"Scenario goal: {scenario_goal}\n"
        f"Question asked: {question}\n"
        f"Improved expression: {better_expression}"
    )
    raw = chat(system, user, max_tokens=64)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return int(json.loads(cleaned).get("score", 85))
    except (json.JSONDecodeError, ValueError):
        return 85


def _generate_turn_reason(
    transcript: str, score: int, better_expression: str, scenario_goal: str
) -> str:
    system = (
        "당신은 영어 학습 피드백 전문가입니다. "
        "유저의 발화에 대해 한국어로 1문장 피드백을 작성하세요. "
        "점수가 낮으면 부족한 점을, 높으면 잘한 점을 간단히 언급하세요."
    )
    user = (
        f"목표: {scenario_goal}\n"
        f"유저 발화: {transcript}\n"
        f"이해도 점수: {score}/100\n"
        f"더 나은 표현: {better_expression}"
    )
    return chat(system, user, max_tokens=128)


def _generate_summary(request: SessionFeedbackRequest) -> str:
    all_text = "\n".join(f"- Q{t.turnIndex}: {t.questionText}\n  A: {t.userTranscript}" for t in request.turns)
    system = (
        "당신은 영어 학습 피드백 전문가입니다. "
        "아래는 영어 학습자가 롤플레이에서 말한 전체 대화입니다. "
        "학습자의 발화(A)만 분석하여 한국어로 2-3문장 종합 피드백을 작성하세요. "
        "다음 세 가지를 포함하세요:\n"
        "1. 전체적인 영어 수준 한 줄 평\n"
        "2. 여러 턴에서 반복적으로 나타난 문법/표현 패턴 (예: 관사 누락, 직역 표현 등)\n"
        "3. 다음 연습에서 집중하면 좋을 한 가지 포인트\n"
        "턴별 세부 피드백은 쓰지 마세요. AI(Q)의 발화는 평가 대상이 아닙니다."
    )
    user = (
        f"시나리오: {request.scenario.title}\n"
        f"결과: {request.scenarioResult}\n"
        f"대화 (Q=AI, A=학습자):\n{all_text}"
    )
    return chat(system, user, max_tokens=256)


def _format_filled_slots(request: SessionFeedbackRequest) -> str:
    if not request.filledSlots:
        return "none"
    return ", ".join(f"{slot.slotKey}={slot.slotValue}" for slot in request.filledSlots)
