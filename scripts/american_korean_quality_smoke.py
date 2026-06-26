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
    args = parser.parse_args()

    result = run_smoke(
        base_url=args.base_url.rstrip("/"),
        output_dir=args.output_dir,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps({
        "fatalIssueCount": result["metadata"]["fatalIssueCount"],
        "reviewNoteCount": result["metadata"]["reviewNoteCount"],
        "caseCount": result["metadata"]["caseCount"],
        "dryRun": result["metadata"]["dryRun"],
        "markdownPath": result["metadata"]["markdownPath"],
        "jsonPath": result["metadata"]["jsonPath"],
    }, ensure_ascii=False, indent=2))
    return 0 if result["metadata"]["fatalIssueCount"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
