# 미국인 한국어 학습자 오류 유형을 라이브 AI 서버에서 점검한다.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://43.202.146.182:8080"
DEFAULT_OUTPUT_DIR = Path("/private/tmp")

PATTERN_KEYS: tuple[str, ...] = (
    "particle_marker",
    "verb_ending_tense",
    "honorific_politeness",
    "word_order_modifier",
    "spacing_word_boundary",
)

SOURCE_NOTES: tuple[dict[str, str], ...] = (
    {
        "title": "Refining Word-Based Grammatical Error Annotation for L2 Korean",
        "url": "https://arxiv.org/abs/2605.30545",
        "reason": "Groups L2 Korean errors around functional morphemes, word boundaries, word order, and spelling.",
    },
    {
        "title": "Korean grammar",
        "url": "https://en.wikipedia.org/wiki/Korean_grammar",
        "reason": "Summarizes Korean postpositions, verb-final structure, particles, and endings.",
    },
    {
        "title": "Korean honorifics and speech levels",
        "url": "https://en.wikipedia.org/wiki/Korean_honorifics",
        "reason": "Grounds politeness and honorific cases as a Korean-specific quality axis.",
    },
)


class KoreanLearnerCase(NamedTuple):
    case_id: str
    pattern_key: str
    purpose: str
    scenario_id: int
    title: str
    briefing: str
    conversation_goal: str
    counterpart_role: str
    ai_question: str
    translated_question: str
    user_utterance: str
    expected_feedback_type: str
    expected_correction_contains: tuple[str, ...]
    expected_benchmark_message: str | None = None


class ScenarioTurn(NamedTuple):
    turn_key: str
    purpose: str
    ai_question: str
    translated_question: str
    user_utterance: str
    expected_feedback_type: str
    expected_correction_contains: tuple[str, ...] = ()
    expected_benchmark_message: str | None = None


class ScenarioQualityCase(NamedTuple):
    case_id: str
    scenario_id: int
    title: str
    briefing: str
    conversation_goal: str
    counterpart_role: str
    turns: tuple[ScenarioTurn, ...]


CASES: tuple[KoreanLearnerCase, ...] = (
    KoreanLearnerCase(
        "AK-PARTICLE-1",
        "particle_marker",
        "Destination particle omission should be corrected with 에.",
        201,
        "친구와 주말 계획 말하기",
        "친구와 주말에 어디에 갈지 이야기합니다.",
        "주말 계획을 한국어로 자연스럽게 말할 수 있다.",
        "friend",
        "주말에 어디에 가요?",
        "Where are you going this weekend?",
        "저는 학교 가요.",
        "NEEDS_IMPROVEMENT",
        ("학교에 가요",),
    ),
    KoreanLearnerCase(
        "AK-PARTICLE-2",
        "particle_marker",
        "Object particle omission should be corrected with 을/를.",
        202,
        "친구와 좋아하는 음식 말하기",
        "친구에게 좋아하는 한국 음식을 말합니다.",
        "좋아하는 음식을 한국어로 자연스럽게 설명할 수 있다.",
        "friend",
        "무슨 음식을 좋아해요?",
        "What food do you like?",
        "저는 김치 좋아해요.",
        "NEEDS_IMPROVEMENT",
        ("김치를 좋아해요",),
    ),
    KoreanLearnerCase(
        "AK-ENDING-1",
        "verb_ending_tense",
        "Past-time question should trigger past tense correction.",
        203,
        "친구와 어제 한 일 말하기",
        "친구에게 어제 한 일을 이야기합니다.",
        "과거 경험을 한국어 시제로 자연스럽게 말할 수 있다.",
        "friend",
        "어제 뭐 했어요?",
        "What did you do yesterday?",
        "어제 영화를 봐요.",
        "NEEDS_IMPROVEMENT",
        ("영화를 봤어요", "영화 봤어요"),
    ),
    KoreanLearnerCase(
        "AK-ENDING-2",
        "verb_ending_tense",
        "Future plan should use future ending rather than present tense.",
        204,
        "친구와 내일 계획 말하기",
        "친구에게 내일 할 일을 이야기합니다.",
        "미래 계획을 한국어로 자연스럽게 말할 수 있다.",
        "friend",
        "내일 뭐 할 거예요?",
        "What will you do tomorrow?",
        "내일 친구를 만나요.",
        "NEEDS_IMPROVEMENT",
        ("친구를 만날 거예요", "친구 만날 거예요"),
    ),
    KoreanLearnerCase(
        "AK-HONORIFIC-1",
        "honorific_politeness",
        "Request to a teacher should be softened with honorific/polite wording.",
        205,
        "선생님께 부탁하기",
        "선생님께 필요한 것을 정중하게 부탁합니다.",
        "상대에 맞는 높임과 공손한 요청을 사용할 수 있다.",
        "teacher",
        "선생님께 무엇을 부탁하고 싶어요?",
        "What would you like to ask your teacher for?",
        "선생님, 물 줘요.",
        "NEEDS_IMPROVEMENT",
        ("물 좀 주시겠어요", "물을 좀 주시겠어요", "물 주세요"),
    ),
    KoreanLearnerCase(
        "AK-HONORIFIC-2",
        "honorific_politeness",
        "Cafe order should avoid blunt command style.",
        206,
        "카페에서 주문하기",
        "카페 직원에게 음료를 주문합니다.",
        "카페에서 공손하게 원하는 음료를 주문할 수 있다.",
        "cafe staff",
        "무엇을 주문하고 싶어요?",
        "What would you like to order?",
        "커피 줘.",
        "NEEDS_IMPROVEMENT",
        ("커피 주세요", "커피를 주세요", "커피 한 잔 주세요"),
    ),
    KoreanLearnerCase(
        "AK-WORDORDER-1",
        "word_order_modifier",
        "English-influenced verb-object order should be corrected.",
        207,
        "친구와 먹는 음식 말하기",
        "친구에게 지금 먹는 음식을 말합니다.",
        "목적어와 서술어의 자연스러운 한국어 어순을 사용할 수 있다.",
        "friend",
        "무엇을 먹어요?",
        "What are you eating?",
        "저는 먹어요 김치를.",
        "NEEDS_IMPROVEMENT",
        ("김치를 먹어요",),
    ),
    KoreanLearnerCase(
        "AK-WORDORDER-2",
        "word_order_modifier",
        "Modifier and object should appear before the verb in natural Korean.",
        208,
        "친구와 읽는 책 말하기",
        "친구에게 읽는 책을 설명합니다.",
        "수식어와 목적어를 자연스러운 한국어 어순으로 말할 수 있다.",
        "friend",
        "어떤 책을 읽어요?",
        "What kind of book are you reading?",
        "저는 읽어요 재미있는 책을.",
        "NEEDS_IMPROVEMENT",
        ("재미있는 책을 읽어요",),
    ),
    KoreanLearnerCase(
        "AK-SPACING-1",
        "spacing_word_boundary",
        "Missing spaces should be corrected while preserving meaning.",
        209,
        "친구와 가는 곳 말하기",
        "친구에게 지금 가는 곳을 말합니다.",
        "한국어 단어 경계를 자연스럽게 띄어 쓸 수 있다.",
        "friend",
        "어디에 가요?",
        "Where are you going?",
        "저는학교에가요.",
        "NEEDS_IMPROVEMENT",
        ("저는 학교에 가요",),
    ),
    KoreanLearnerCase(
        "AK-SPACING-2",
        "spacing_word_boundary",
        "Spacing plus object marker should be corrected in a food preference answer.",
        210,
        "친구와 한국 음식 말하기",
        "친구에게 좋아하는 한국 음식을 말합니다.",
        "띄어쓰기와 조사까지 포함해 자연스럽게 답할 수 있다.",
        "friend",
        "무슨 한국 음식을 좋아해요?",
        "What Korean food do you like?",
        "저는한국음식좋아해요.",
        "NEEDS_IMPROVEMENT",
        ("저는 한국 음식을 좋아해요", "한국 음식을 좋아해요"),
    ),
)


SCENARIO_CASES: tuple[ScenarioQualityCase, ...] = (
    ScenarioQualityCase(
        "SQ-FANSIGN",
        301,
        "영상통화 팬사인회",
        "최애와 영상통화 팬사인회에서 짧게 대화합니다.",
        "좋아하는 아이돌에게 자연스럽고 호감 있는 한국어로 답할 수 있다.",
        "idol at a video fan-sign event",
        (
            ScenarioTurn(
                "Q1",
                "Long-time-no-see greeting should be answered warmly.",
                "안녕~ 엄청 오랜만이네! 보니까 너무 좋다. 잘 지냈어?",
                "Hey~ It's been so long! It's so good to see you. How've you been?",
                "네, 잘 지냈어요!",
                "GOOD",
            ),
            ScenarioTurn(
                "Q2",
                "Directly accepting a compliment should be softened with modest fan-sign wording.",
                "한국어가 더 는 것 같은데? 어떻게 이렇게 잘해?",
                "Your Korean's gotten even better, hasn't it? How are you this good at it?",
                "네, 저 한국어 잘해요.",
                "NEEDS_IMPROVEMENT",
                ("아직 부족", "열심히 공부", "아직 많이 부족"),
            ),
            ScenarioTurn(
                "Q3",
                "Favorite-song answer should give a clear reason.",
                "우리 노래 중에는 뭐가 제일 좋아? 왜 그 노래야?",
                "Which of our songs is your favorite? Why that one?",
                "저는 별빛 노래가 제일 좋아요. 가사가 예뻐서요.",
                "GOOD",
            ),
            ScenarioTurn(
                "Q4",
                "Saying 상관없어요 to a final fan-sign prompt sounds uninterested.",
                "마지막으로 나한테 하고 싶은 말 있어? 뭐든!",
                "Anything you wanna say to me before we wrap up? Anything at all!",
                "상관없어요.",
                "NEEDS_IMPROVEMENT",
                ("만나서 정말 좋", "항상 응원", "만나서 좋"),
            ),
        ),
    ),
    ScenarioQualityCase(
        "SQ-FAN-FRIEND",
        302,
        "같은 그룹 덕메랑 1:1 대화",
        "같은 그룹을 좋아하는 또래 팬과 친근하게 대화합니다.",
        "덕메와 너무 딱딱하지 않은 반말 톤으로 취향과 계획을 나눌 수 있다.",
        "same-age K-pop fan friend",
        (
            ScenarioTurn(
                "Q1",
                "Overly formal bias answer should become casual.",
                "안녕! 어 너도 이 그룹 좋아해? 나도야! 너 최애 누구야?",
                "Hi! Oh, you like this group too? Me too! Who's your bias?",
                "제 최애는 민지입니다.",
                "NEEDS_IMPROVEMENT",
                ("내 최애는 민지야", "나는 민지가 최애야"),
            ),
            ScenarioTurn(
                "Q2",
                "Report-style origin story should become casual fan talk.",
                "헐 우리 취향 비슷하다! 어쩌다 입덕했어?",
                "OMG we have similar taste! How'd you get into them?",
                "저는 유튜브에서 무대를 보고 입덕했습니다.",
                "NEEDS_IMPROVEMENT",
                ("무대 보고 입덕했어", "유튜브에서"),
            ),
            ScenarioTurn(
                "Q3",
                "Formal concert experience answer should become casual.",
                "너 오프라인 콘서트나 팬싸 가본 적 있어? 아님 나중에 갈 생각 있어?",
                "Have you ever been to a concert or fan-sign in person? Or are you thinking of going sometime?",
                "아직 안 가봤습니다. 나중에 가고 싶습니다.",
                "NEEDS_IMPROVEMENT",
                ("아직 안 가봤어", "가고 싶어"),
            ),
            ScenarioTurn(
                "Q4",
                "Formal acceptance to a friendly concert invitation should become casual.",
                "야 우리 완전 잘 맞는다. 다음에 콘서트 같이 갈래? 나 같은 그룹 덕질하는 친구 없어 ㅠㅠ",
                "Hey, we totally click. Wanna go to a concert together sometime? I don't have any friends who stan the same group.",
                "네, 같이 가고 싶습니다.",
                "NEEDS_IMPROVEMENT",
                ("응", "같이 가고 싶어", "같이 가자"),
            ),
        ),
    ),
    ScenarioQualityCase(
        "SQ-DATE",
        303,
        "한국인과 소개팅",
        "지인 소개로 만난 한국인과 첫 소개팅에서 대화합니다.",
        "존댓말은 유지하되 딱딱하거나 직설적인 표현을 피하고 호감 있는 톤으로 말할 수 있다.",
        "Korean blind date partner",
        (
            ScenarioTurn(
                "Q1",
                "아무거나요 can sound passive or uninterested on a first date.",
                "안녕하세요! 만나서 반갑습니다 ㅎㅎ 뭐 드시고 싶으세요? 좋아하는 음식이 뭐예요?",
                "Hi, nice to meet you hehe. What would you like to eat? What kind of food do you like?",
                "아무거나요.",
                "NEEDS_IMPROVEMENT",
                ("좋아해요", "괜찮으세요", "먹고 싶어요"),
            ),
            ScenarioTurn(
                "Q2",
                "Report-style weekend answer should sound warmer and more conversational.",
                "주말엔 보통 뭐 하면서 시간 보내세요?",
                "What do you usually do on weekends?",
                "저는 주말에 집에서 휴식을 취합니다.",
                "NEEDS_IMPROVEMENT",
                ("집에서 쉬", "영화", "산책"),
            ),
            ScenarioTurn(
                "Q3",
                "Ideal-type answer should avoid shallow one-word attraction and add personality.",
                "혹시 이상형 물어봐도 돼요? 이상형이 어떻게 돼요?",
                "Can I ask what your type is? What's your ideal type?",
                "예쁜 사람이 좋아요.",
                "NEEDS_IMPROVEMENT",
                ("대화가 잘 통", "편안한", "성격"),
            ),
            ScenarioTurn(
                "Q4_ACCEPT",
                "Direct cool acceptance sounds too forward for a first meeting.",
                "오늘 대화 너무 재밌었어요. 집까지 데려다드릴까요?",
                "I had such a great time today. Can I give you a ride home?",
                "당연하죠!",
                "NEEDS_IMPROVEMENT",
                ("그래도 될까요", "감사", "고마워요"),
            ),
            ScenarioTurn(
                "Q4_REJECT",
                "Direct refusal should use a cushion phrase for offered help.",
                "오늘 대화 너무 재밌었어요. 집까지 데려다드릴까요?",
                "I had such a great time today. Can I give you a ride home?",
                "아니요, 싫어요.",
                "NEEDS_IMPROVEMENT",
                ("감사하지만", "괜찮아요", "혼자 갈게요"),
            ),
        ),
    ),
)


def build_turn_feedback_payload(
    case: KoreanLearnerCase,
    *,
    session_id: int,
    turn_id: int,
) -> dict[str, Any]:
    return {
        "sessionId": session_id,
        "turnId": turn_id,
        "sequence": 1,
        "scenario": _scenario_payload(case),
        "turn": {
            "aiQuestion": case.ai_question,
            "translatedQuestion": case.translated_question,
            "userUtterance": case.user_utterance,
        },
    }


def build_session_feedback_payload(
    case: KoreanLearnerCase,
    *,
    session_id: int,
    expected_turn_ids: list[int],
) -> dict[str, Any]:
    return {
        "sessionId": session_id,
        "scenario": _scenario_payload(case),
        "expectedTurnIds": expected_turn_ids,
    }


def _scenario_payload(case: KoreanLearnerCase) -> dict[str, Any]:
    return {
        "scenarioId": case.scenario_id,
        "title": case.title,
        "briefing": case.briefing,
        "conversationGoal": case.conversation_goal,
        "counterpartRole": case.counterpart_role,
        "serviceAudience": "AMERICAN_LEARNER",
    }


def build_scenario_turn_feedback_payload(
    case: ScenarioQualityCase,
    turn: ScenarioTurn,
    *,
    session_id: int,
    turn_id: int,
) -> dict[str, Any]:
    return {
        "sessionId": session_id,
        "turnId": turn_id,
        "sequence": _turn_sequence(turn),
        "scenario": _scenario_quality_payload(case),
        "turn": {
            "aiQuestion": turn.ai_question,
            "translatedQuestion": turn.translated_question,
            "userUtterance": turn.user_utterance,
        },
    }


def build_scenario_next_question_payload(
    case: ScenarioQualityCase,
    *,
    turn_index: int,
    session_id: int,
    turn_id: int,
) -> dict[str, Any]:
    current_turn = case.turns[turn_index]
    next_turn = case.turns[turn_index + 1]
    return {
        "sessionId": session_id,
        "submittedTurnId": turn_id,
        "submittedSequence": _turn_sequence(current_turn),
        "scenario": _scenario_quality_payload(case),
        "currentTurn": {
            "aiQuestion": current_turn.ai_question,
            "translatedQuestion": current_turn.translated_question,
            "userUtterance": current_turn.user_utterance,
        },
        "nextQuestion": {
            "questionId": case.scenario_id * 100 + _turn_sequence(next_turn),
            "sequence": _turn_sequence(next_turn),
            "questionEn": next_turn.translated_question,
            "questionKo": next_turn.ai_question,
        },
    }


def build_scenario_closing_message_payload(
    case: ScenarioQualityCase,
    turn: ScenarioTurn,
    *,
    session_id: int,
    turn_id: int,
) -> dict[str, Any]:
    return {
        "sessionId": session_id,
        "submittedTurnId": turn_id,
        "submittedSequence": _turn_sequence(turn),
        "scenario": _scenario_quality_payload(case),
        "currentTurn": {
            "aiQuestion": turn.ai_question,
            "translatedQuestion": turn.translated_question,
            "userUtterance": turn.user_utterance,
        },
        "closingReason": "GOAL_COMPLETED",
        "goalCompletionStatus": "COMPLETED",
    }


def build_scenario_session_feedback_payload(
    case: ScenarioQualityCase,
    *,
    session_id: int,
    expected_turn_ids: list[int],
) -> dict[str, Any]:
    return {
        "sessionId": session_id,
        "scenario": _scenario_quality_payload(case),
        "expectedTurnIds": expected_turn_ids,
    }


def _scenario_quality_payload(case: ScenarioQualityCase) -> dict[str, Any]:
    return {
        "scenarioId": case.scenario_id,
        "title": case.title,
        "briefing": case.briefing,
        "conversationGoal": case.conversation_goal,
        "counterpartRole": case.counterpart_role,
        "serviceAudience": "AMERICAN_LEARNER",
    }


def _turn_sequence(turn: ScenarioTurn) -> int:
    if turn.turn_key.startswith("Q4"):
        return 4
    return int(turn.turn_key.removeprefix("Q"))


def expected_output(case: KoreanLearnerCase) -> dict[str, Any]:
    return {
        "feedbackType": case.expected_feedback_type,
        "patternKey": case.pattern_key,
        "benchmarkMessage": case.expected_benchmark_message,
        "correctionExpressionContainsAny": list(case.expected_correction_contains),
        "correctionExpressionLanguage": "Korean",
        "feedbackDetail": None,
        "correctionReasonLanguage": "English",
    }


def expected_scenario_turn_output(turn: ScenarioTurn) -> dict[str, Any]:
    return {
        "feedbackType": turn.expected_feedback_type,
        "benchmarkMessage": turn.expected_benchmark_message,
        "correctionExpressionContainsAny": list(turn.expected_correction_contains),
        "correctionExpressionLanguage": "Korean",
        "correctionReasonLanguage": "English",
    }


def run_smoke(
    *,
    base_url: str,
    output_dir: Path,
    limit: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_cases = list(CASES[: limit or None])
    generated_at = datetime.now(timezone.utc)
    session_base = int(time.time()) % 1_000_000 + 1_700_000
    result: dict[str, Any] = {
        "metadata": {
            "executedAt": generated_at.isoformat(),
            "baseUrl": base_url,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "caseCount": len(selected_cases),
            "dryRun": dry_run,
            "sources": list(SOURCE_NOTES),
        },
        "cases": [],
        "fatalIssues": [],
        "reviewNotes": [],
    }

    if not dry_run:
        result["metadata"]["health"] = _request_json("GET", f"{base_url}/health", None)

    for index, case in enumerate(selected_cases, start=1):
        session_id = session_base + index
        turn_id = session_id * 1000 + index
        turn_payload = build_turn_feedback_payload(case, session_id=session_id, turn_id=turn_id)
        session_payload = build_session_feedback_payload(case, session_id=session_id, expected_turn_ids=[turn_id])
        actual_output = None
        elapsed_ms = None
        fatal_issues: list[str] = []
        review_notes: list[str] = []

        if dry_run:
            review_notes.append("dry-run이라 실제 API output은 생성하지 않음")
        else:
            started = time.perf_counter()
            turn_feedback_creation = _request_json(
                "POST",
                f"{base_url}/api/v1/conversation/turn-feedback",
                turn_payload,
            )
            session_feedback = _request_json(
                "POST",
                f"{base_url}/api/v1/conversation/session-feedback",
                session_payload,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            turn_feedback = _turn_feedback_by_id(session_feedback, turn_id)
            actual_output = {
                "turnFeedbackCreation": turn_feedback_creation,
                "sessionFeedback": session_feedback,
                "turnFeedback": turn_feedback,
            }
            fatal_issues, review_notes = evaluate_actual_output(case, actual_output)

        case_result = {
            "caseId": case.case_id,
            "patternKey": case.pattern_key,
            "purpose": case.purpose,
            "input": {
                "turnFeedbackRequest": turn_payload,
                "sessionFeedbackRequest": session_payload,
            },
            "expected": expected_output(case),
            "actualOutput": actual_output,
            "elapsedMs": elapsed_ms,
            "fatalIssues": fatal_issues,
            "reviewNotes": review_notes,
        }
        result["cases"].append(case_result)
        for issue in fatal_issues:
            result["fatalIssues"].append(f"{case.case_id} | {issue}")
        for note in review_notes:
            result["reviewNotes"].append(f"{case.case_id} | {note}")

    result["metadata"]["fatalIssueCount"] = len(result["fatalIssues"])
    result["metadata"]["reviewNoteCount"] = len(result["reviewNotes"])
    result["metadata"]["finishedAt"] = datetime.now(timezone.utc).isoformat()

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_json_path = output_dir / f"saynow-american-korean-quality-smoke-{stamp}.json"
    output_md_path = output_dir / f"saynow-american-korean-quality-smoke-{stamp}.md"
    output_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md_path.write_text(render_markdown_report(result, output_json_path), encoding="utf-8")
    result["metadata"]["jsonPath"] = str(output_json_path)
    result["metadata"]["markdownPath"] = str(output_md_path)
    return result


def run_scenario_smoke(
    *,
    base_url: str,
    output_dir: Path,
    limit: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_cases = list(SCENARIO_CASES[: limit or None])
    generated_at = datetime.now(timezone.utc)
    session_base = int(time.time()) % 1_000_000 + 1_900_000
    result: dict[str, Any] = {
        "metadata": {
            "executedAt": generated_at.isoformat(),
            "baseUrl": base_url,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "scenarioCount": len(selected_cases),
            "turnCaseCount": sum(len(case.turns) for case in selected_cases),
            "dryRun": dry_run,
        },
        "scenarios": [],
        "fatalIssues": [],
        "reviewNotes": [],
    }

    if not dry_run:
        result["metadata"]["health"] = _request_json("GET", f"{base_url}/health", None)

    for scenario_index, case in enumerate(selected_cases, start=1):
        session_id = session_base + scenario_index
        turn_results: list[dict[str, Any]] = []
        expected_turn_ids: list[int] = []
        scenario_fatal: list[str] = []
        scenario_review: list[str] = []

        for turn_index, turn in enumerate(case.turns):
            turn_id = session_id * 1000 + turn_index + 1
            expected_turn_ids.append(turn_id)
            turn_feedback_payload = build_scenario_turn_feedback_payload(
                case,
                turn,
                session_id=session_id,
                turn_id=turn_id,
            )
            conversation_payload = _build_scenario_conversation_payload(
                case,
                turn,
                turn_index=turn_index,
                session_id=session_id,
                turn_id=turn_id,
            )
            turn_result = {
                "turnKey": turn.turn_key,
                "purpose": turn.purpose,
                "input": {
                    "conversationRequest": conversation_payload,
                    "turnFeedbackRequest": turn_feedback_payload,
                },
                "expected": expected_scenario_turn_output(turn),
                "actualOutput": None,
                "elapsedMs": None,
                "fatalIssues": [],
                "reviewNotes": [],
            }

            if dry_run:
                turn_result["reviewNotes"].append("dry-run이라 실제 API output은 생성하지 않음")
            else:
                started = time.perf_counter()
                conversation_output = _request_json(
                    "POST",
                    _scenario_conversation_endpoint(base_url, turn),
                    conversation_payload,
                )
                turn_feedback_creation = _request_json(
                    "POST",
                    f"{base_url}/api/v1/conversation/turn-feedback",
                    turn_feedback_payload,
                )
                turn_result["elapsedMs"] = round((time.perf_counter() - started) * 1000, 2)
                turn_result["actualOutput"] = {
                    "conversation": conversation_output,
                    "turnFeedbackCreation": turn_feedback_creation,
                    "turnFeedback": None,
                }

            turn_results.append(turn_result)

        session_payload = build_scenario_session_feedback_payload(
            case,
            session_id=session_id,
            expected_turn_ids=expected_turn_ids,
        )
        session_feedback = None
        if not dry_run:
            session_feedback = _request_json(
                "POST",
                f"{base_url}/api/v1/conversation/session-feedback",
                session_payload,
            )
            for turn_result, turn_id in zip(turn_results, expected_turn_ids, strict=True):
                if isinstance(turn_result.get("actualOutput"), dict):
                    turn_result["actualOutput"]["turnFeedback"] = _turn_feedback_by_id(session_feedback, turn_id)

        for turn, turn_result in zip(case.turns, turn_results, strict=True):
            if dry_run:
                fatal_issues: list[str] = []
                review_notes = list(turn_result["reviewNotes"])
            else:
                fatal_issues, review_notes = evaluate_scenario_turn_output(turn, turn_result["actualOutput"])
            turn_result["fatalIssues"] = fatal_issues
            turn_result["reviewNotes"] = review_notes
            for issue in fatal_issues:
                scenario_fatal.append(f"{turn.turn_key} | {issue}")
            for note in review_notes:
                scenario_review.append(f"{turn.turn_key} | {note}")

        scenario_result = {
            "caseId": case.case_id,
            "title": case.title,
            "scenario": _scenario_quality_payload(case),
            "sessionFeedbackRequest": session_payload,
            "sessionFeedback": session_feedback,
            "turns": turn_results,
            "fatalIssues": scenario_fatal,
            "reviewNotes": scenario_review,
        }
        result["scenarios"].append(scenario_result)
        for issue in scenario_fatal:
            result["fatalIssues"].append(f"{case.case_id} | {issue}")
        for note in scenario_review:
            result["reviewNotes"].append(f"{case.case_id} | {note}")

    result["metadata"]["fatalIssueCount"] = len(result["fatalIssues"])
    result["metadata"]["reviewNoteCount"] = len(result["reviewNotes"])
    result["metadata"]["finishedAt"] = datetime.now(timezone.utc).isoformat()

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    output_json_path = output_dir / f"saynow-american-korean-scenario-quality-smoke-{stamp}.json"
    output_md_path = output_dir / f"saynow-american-korean-scenario-quality-smoke-{stamp}.md"
    output_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md_path.write_text(render_scenario_markdown_report(result, output_json_path), encoding="utf-8")
    result["metadata"]["jsonPath"] = str(output_json_path)
    result["metadata"]["markdownPath"] = str(output_md_path)
    return result


def _build_scenario_conversation_payload(
    case: ScenarioQualityCase,
    turn: ScenarioTurn,
    *,
    turn_index: int,
    session_id: int,
    turn_id: int,
) -> dict[str, Any]:
    if _is_closing_turn(turn):
        return build_scenario_closing_message_payload(
            case,
            turn,
            session_id=session_id,
            turn_id=turn_id,
        )
    return build_scenario_next_question_payload(
        case,
        turn_index=turn_index,
        session_id=session_id,
        turn_id=turn_id,
    )


def _scenario_conversation_endpoint(base_url: str, turn: ScenarioTurn) -> str:
    if _is_closing_turn(turn):
        return f"{base_url}/api/v1/conversation/closing-message"
    return f"{base_url}/api/v1/conversation/next-question"


def _is_closing_turn(turn: ScenarioTurn) -> bool:
    return turn.turn_key.startswith("Q4")


def evaluate_actual_output(case: KoreanLearnerCase, actual_output: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    fatal: list[str] = []
    review: list[str] = []
    if actual_output is None:
        return ["actual output이 없음"], review
    feedback_output = _extract_turn_feedback_output(actual_output)
    if feedback_output is None:
        return ["actual output에 turn feedback이 없음"], review

    feedback_type = feedback_output.get("feedbackType")
    if feedback_type != case.expected_feedback_type:
        fatal.append(f"feedbackType 기대={case.expected_feedback_type}, 실제={feedback_type}")

    if feedback_output.get("benchmarkMessage") is not None:
        fatal.append("AMERICAN_LEARNER인데 benchmarkMessage가 null이 아님")

    if feedback_type == "NEEDS_IMPROVEMENT":
        if not feedback_output.get("positiveFeedback"):
            fatal.append("NEEDS_IMPROVEMENT인데 positiveFeedback이 없음")
        if feedback_output.get("feedbackDetail") is not None:
            fatal.append("NEEDS_IMPROVEMENT인데 feedbackDetail이 null이 아님")
        correction = str(feedback_output.get("correctionExpression") or "")
        reason = str(feedback_output.get("correctionReason") or "")
        if not correction:
            fatal.append("NEEDS_IMPROVEMENT인데 correctionExpression이 없음")
        if not reason:
            fatal.append("NEEDS_IMPROVEMENT인데 correctionReason이 없음")
        if correction and not any(fragment in correction for fragment in case.expected_correction_contains):
            review.append(
                "correctionExpression이 기대 fragment와 다름. "
                f"expected_any={case.expected_correction_contains}, actual={correction}"
            )
        if correction and not _contains_hangul(correction):
            fatal.append("correctionExpression이 한국어로 보이지 않음")
    return fatal, review


def evaluate_scenario_turn_output(
    turn: ScenarioTurn,
    actual_output: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    fatal: list[str] = []
    review: list[str] = []
    if actual_output is None:
        return ["actual output이 없음"], review

    conversation_output = actual_output.get("conversation")
    if isinstance(conversation_output, dict):
        inner_thought = str(conversation_output.get("innerThought") or "")
        if not inner_thought:
            fatal.append("conversation output에 innerThought가 없음")
        if inner_thought and _contains_hangul(inner_thought):
            fatal.append("AMERICAN_LEARNER인데 conversation innerThought가 영어가 아님")
        if _has_planner_inner_thought_marker(inner_thought):
            fatal.append("conversation innerThought가 다음 행동/진행 계획처럼 보임")
    else:
        fatal.append("conversation output이 없음")

    feedback_output = _extract_turn_feedback_output(actual_output)
    if feedback_output is None:
        fatal.append("actual output에 turn feedback이 없음")
        return fatal, review

    feedback_type = feedback_output.get("feedbackType")
    if feedback_type != turn.expected_feedback_type:
        fatal.append(f"feedbackType 기대={turn.expected_feedback_type}, 실제={feedback_type}")

    if feedback_output.get("benchmarkMessage") is not None:
        fatal.append("AMERICAN_LEARNER인데 benchmarkMessage가 null이 아님")

    if feedback_type == "NEEDS_IMPROVEMENT":
        if not feedback_output.get("positiveFeedback"):
            fatal.append("NEEDS_IMPROVEMENT인데 positiveFeedback이 없음")
        if feedback_output.get("feedbackDetail") is not None:
            fatal.append("NEEDS_IMPROVEMENT인데 feedbackDetail이 null이 아님")
        correction = str(feedback_output.get("correctionExpression") or "")
        reason = str(feedback_output.get("correctionReason") or "")
        if not correction:
            fatal.append("NEEDS_IMPROVEMENT인데 correctionExpression이 없음")
        if not reason:
            fatal.append("NEEDS_IMPROVEMENT인데 correctionReason이 없음")
        if correction and not _contains_hangul(correction):
            fatal.append("correctionExpression이 한국어로 보이지 않음")
        if reason and _contains_hangul(reason):
            review.append("correctionReason에 한국어가 섞임")
        if correction and turn.expected_correction_contains and not any(
            fragment in correction for fragment in turn.expected_correction_contains
        ):
            review.append(
                "correctionExpression이 기대 fragment와 다름. "
                f"expected_any={turn.expected_correction_contains}, actual={correction}"
            )
    elif feedback_type == "GOOD":
        if feedback_output.get("correctionExpression") is not None:
            fatal.append("GOOD인데 correctionExpression이 null이 아님")
        if feedback_output.get("feedbackDetail") is None:
            fatal.append("GOOD인데 feedbackDetail이 없음")
    return fatal, review


def _extract_turn_feedback_output(actual_output: dict[str, Any]) -> dict[str, Any] | None:
    turn_feedback = actual_output.get("turnFeedback")
    if isinstance(turn_feedback, dict):
        return turn_feedback
    if "feedbackType" in actual_output:
        return actual_output
    return None


def _turn_feedback_by_id(session_feedback: dict[str, Any], turn_id: int) -> dict[str, Any] | None:
    for feedback in session_feedback.get("turnFeedbacks", []):
        if feedback.get("turnId") == turn_id:
            return feedback
    return None


def render_markdown_report(result: dict[str, Any], output_json_path: Path) -> str:
    lines = [
        "# American Learner Korean Quality Smoke",
        "",
        f"- 실행 시각: `{result['metadata']['executedAt']}`",
        f"- Base URL: `{result['metadata']['baseUrl']}`",
        f"- Branch: `{result['metadata'].get('branch', 'unknown')}`",
        f"- Commit: `{result['metadata'].get('commit', 'unknown')}`",
        f"- Dry run: `{result['metadata'].get('dryRun', False)}`",
        f"- 원본 JSON: `{output_json_path}`",
        "",
        "## Summary",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| case | {result['metadata']['caseCount']} |",
        f"| fatal issue | {result['metadata'].get('fatalIssueCount', 0)} |",
        f"| review note | {result['metadata'].get('reviewNoteCount', 0)} |",
        "",
        "## Sources",
        "",
    ]
    for source in result["metadata"].get("sources", []):
        lines.append(f"- [{source['title']}]({source['url']}) - {source['reason']}")

    lines.extend(["", "## Fatal Issues", ""])
    if result.get("fatalIssues"):
        lines.extend(f"- {issue}" for issue in result["fatalIssues"])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Review Notes", ""])
    if result.get("reviewNotes"):
        lines.extend(f"- {note}" for note in result["reviewNotes"])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Cases", ""])
    for index, case_result in enumerate(result["cases"], start=1):
        lines.extend([
            f"## Case {index}. {case_result['caseId']} `{case_result['patternKey']}`",
            "",
            f"- 목적: {case_result['purpose']}",
            f"- 판정: `{_case_verdict(case_result)}`",
            "",
            "### Input",
            "",
            "```json",
            _json_block(case_result["input"]),
            "```",
            "",
            "### Expected",
            "",
            "```json",
            _json_block(case_result["expected"]),
            "```",
            "",
            "### Actual Output",
            "",
            "```json",
            _json_block(case_result["actualOutput"]),
            "```",
            "",
            "### Issues",
            "",
        ])
        if case_result["fatalIssues"]:
            lines.extend(f"- FATAL: {issue}" for issue in case_result["fatalIssues"])
        if case_result["reviewNotes"]:
            lines.extend(f"- REVIEW: {note}" for note in case_result["reviewNotes"])
        if not case_result["fatalIssues"] and not case_result["reviewNotes"]:
            lines.append("- 없음")
        lines.append("")
    return "\n".join(lines)


def render_scenario_markdown_report(result: dict[str, Any], output_json_path: Path) -> str:
    lines = [
        "# American Learner Scenario Quality Smoke",
        "",
        f"- 실행 시각: `{result['metadata']['executedAt']}`",
        f"- Base URL: `{result['metadata']['baseUrl']}`",
        f"- Branch: `{result['metadata'].get('branch', 'unknown')}`",
        f"- Commit: `{result['metadata'].get('commit', 'unknown')}`",
        f"- Dry run: `{result['metadata'].get('dryRun', False)}`",
        f"- 원본 JSON: `{output_json_path}`",
        "",
        "## Summary",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| scenario | {result['metadata']['scenarioCount']} |",
        f"| turn case | {result['metadata']['turnCaseCount']} |",
        f"| fatal issue | {result['metadata'].get('fatalIssueCount', 0)} |",
        f"| review note | {result['metadata'].get('reviewNoteCount', 0)} |",
        "",
        "## Fatal Issues",
        "",
    ]
    if result.get("fatalIssues"):
        lines.extend(f"- {issue}" for issue in result["fatalIssues"])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Review Notes", ""])
    if result.get("reviewNotes"):
        lines.extend(f"- {note}" for note in result["reviewNotes"])
    else:
        lines.append("- 없음")

    lines.extend(["", "## Scenarios", ""])
    for scenario_index, scenario_result in enumerate(result["scenarios"], start=1):
        lines.extend([
            f"## Scenario {scenario_index}. {scenario_result['caseId']} - {scenario_result['title']}",
            "",
            "### Scenario Payload",
            "",
            "```json",
            _json_block(scenario_result["scenario"]),
            "```",
            "",
            "### Session Feedback",
            "",
            "#### Input",
            "",
            "```json",
            _json_block(scenario_result.get("sessionFeedbackRequest")),
            "```",
            "",
            "#### Actual Output",
            "",
            "```json",
            _json_block(scenario_result.get("sessionFeedback")),
            "```",
            "",
        ])
        for turn_result in scenario_result["turns"]:
            lines.extend([
                f"### Turn {turn_result['turnKey']}",
                "",
                f"- 목적: {turn_result['purpose']}",
                f"- 판정: `{_case_verdict(turn_result)}`",
                "",
                "### Input",
                "",
                "```json",
                _json_block(turn_result["input"]),
                "```",
                "",
                "### Expected",
                "",
                "```json",
                _json_block(turn_result["expected"]),
                "```",
                "",
                "### Actual Output",
                "",
                "```json",
                _json_block(turn_result["actualOutput"]),
                "```",
                "",
                "### Issues",
                "",
            ])
            if turn_result["fatalIssues"]:
                lines.extend(f"- FATAL: {issue}" for issue in turn_result["fatalIssues"])
            if turn_result["reviewNotes"]:
                lines.extend(f"- REVIEW: {note}" for note in turn_result["reviewNotes"])
            if not turn_result["fatalIssues"] and not turn_result["reviewNotes"]:
                lines.append("- 없음")
            lines.append("")
    return "\n".join(lines)


def _request_json(method: str, url: str, payload: dict[str, Any] | None) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {raw}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _case_verdict(case_result: dict[str, Any]) -> str:
    if case_result["fatalIssues"]:
        return "FAIL"
    if case_result["reviewNotes"]:
        return "REVIEW"
    return "PASS"


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _contains_hangul(value: str) -> bool:
    return re.search(r"[가-힣]", value) is not None


def _has_planner_inner_thought_marker(value: str) -> bool:
    normalized = _normalize_english_text(value)
    markers = [
        "i should ask",
        "i should keep",
        "ask about",
        "ask them about",
        "conversation moving",
        "move the conversation",
        "next topic",
        "wrap up",
        "end the conversation",
    ]
    return any(marker in normalized for marker in markers)


def _normalize_english_text(value: str) -> str:
    lowered = value.lower().strip()
    no_punctuation = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", no_punctuation).strip()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scenario-quality", action="store_true")
    args = parser.parse_args()

    runner = run_scenario_smoke if args.scenario_quality else run_smoke
    result = runner(
        base_url=args.base_url.rstrip("/"),
        output_dir=args.output_dir,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    case_count_key = "turnCaseCount" if args.scenario_quality else "caseCount"
    print(json.dumps({
        "fatalIssueCount": result["metadata"]["fatalIssueCount"],
        "reviewNoteCount": result["metadata"]["reviewNoteCount"],
        "caseCount": result["metadata"][case_count_key],
        "dryRun": result["metadata"]["dryRun"],
        "markdownPath": result["metadata"]["markdownPath"],
        "jsonPath": result["metadata"]["jsonPath"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["metadata"]["fatalIssueCount"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
