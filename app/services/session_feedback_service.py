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

    for i, turn in enumerate(request.turns):
        analysis = _analyze_utterance(turn.transcript, request.scenarioGoal, turn.question)
        score = analysis["comprehension_score"]
        score_delta = score - prev_score if i > 0 else 0
        improved_score = _estimate_improved_score(
            analysis["better_expression"], request.scenarioGoal, turn.question
        )
        reason = _generate_turn_reason(
            turn.transcript, score, analysis["better_expression"], request.scenarioGoal
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
    summary = _generate_summary(request.turns, request.scenarioGoal)

    return SessionFeedbackResponse(
        totalUnderstoodScore=total,
        summary=summary,
        turns=turn_feedbacks,
    )


def _analyze_utterance(transcript: str, scenario_goal: str, question: str) -> dict:
    system = (
        "You are an English language expert analyzing how well a non-native speaker communicated. "
        "Respond ONLY with valid JSON matching this schema exactly:\n"
        '{"comprehension_score": <0-100 int>, "native_perception": "<string>", "better_expression": "<string>"}\n'
        "comprehension_score: how well a native American English speaker would understand the reply (0=not at all, 100=perfectly).\n"
        "native_perception: a short phrase describing what the native speaker actually heard/understood.\n"
        "better_expression: one improved alternative sentence, more natural and clear for a native speaker."
    )
    user = (
        f"Scenario goal: {scenario_goal}\n"
        f"Question asked: {question}\n"
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


def _generate_summary(turns, scenario_goal: str) -> str:
    all_text = "\n".join(f"- {t.transcript}" for t in turns)
    system = (
        "당신은 영어 학습 피드백 전문가입니다. "
        "전체 대화를 보고 한국어로 2-3문장 종합 피드백을 작성하세요. "
        "전반적인 의사소통 수준, 잘한 점, 개선할 점을 포함하세요."
    )
    user = (
        f"목표: {scenario_goal}\n"
        f"유저 발화 목록:\n{all_text}"
    )
    return chat(system, user, max_tokens=256)
