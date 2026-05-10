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
            analysis["better_expression"], scenario_goal, turn.questionText, score
        )
        reason = _generate_turn_reason(
            turn.userTranscript, score, analysis["better_expression"], turn.questionText
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
        "better_expression: one improved alternative that is slightly more natural — aim for a small, achievable improvement (5-10 points better, never exceeding 100), not a perfect rewrite. Keep it close to the user's original phrasing and level."
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


def _estimate_improved_score(better_expression: str, scenario_goal: str, question: str, original_score: int) -> int:
    system = (
        "You are an English language expert. "
        "If the user had said the following improved expression, what comprehension_score (0-100) "
        "would a native American English speaker give? "
        f"The score MUST be between {min(100, original_score + 5)} and {min(100, original_score + 15)}. "
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
        result = int(json.loads(cleaned).get("score", original_score + 5))
        return min(100, original_score + 15, max(original_score + 5, result))
    except (json.JSONDecodeError, ValueError):
        return min(100, original_score + 5)


def _generate_turn_reason(
    transcript: str, score: int, better_expression: str, question: str
) -> str:
    system = (
        "당신은 영어 학습 피드백 전문가입니다. "
        "아래 발화는 음성을 STT로 변환한 텍스트입니다. "
        "유저의 발화에 대해 한국어로 1문장 피드백을 작성하세요. "
        "평가 기준은 다음과 같습니다:\n"
        "- 문법 오류 (관사 누락, 시제, 어순, 복수형 등)\n"
        "- 표현의 자연스러움 (직역투, 어색한 단어 선택 등)\n"
        "- 맥락 적절성 (질문에 맞게 대답했는지)\n"
        "절대 평가하지 말아야 할 것: 대문자/소문자, 구두점, 철자 — 이것들은 음성에 존재하지 않습니다. "
        "내용이 맞는지(시나리오 목표 달성 여부)는 평가하지 마세요. 어떻게 말했는지만 평가하세요. "
        "개선 사항을 제시할 때는, 반드시 유저의 실제 표현과 개선된 표현을 구체적으로 언급하세요. "
        "예: '\"I want\" 대신 \"I'd like\"을 쓰면 더 자연스럽습니다.' 처럼 구체적인 표현을 명시하세요."
    )
    user = (
        f"AI 질문: {question}\n"
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
        "학습자의 발화(A)만 분석하여 한국어로 2-3문장 줄글로 종합 피드백을 작성하세요. "
        "번호나 글머리 기호 없이 자연스러운 문장으로 작성하세요. "
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
