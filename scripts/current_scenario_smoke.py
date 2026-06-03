# 3차 MVP 현재 시나리오 데이터로 AI 서버 품질을 재검증한다.
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ATTACHMENT_PATH = Path("/Users/sangmin8817/.codex/attachments/049d2a95-1263-4ab8-b1e0-e2a8ceabbb13/pasted-text.txt")
PREVIOUS_BASELINE_PATH = Path("/private/tmp/saynow_3mvp_current_scenario_quality_smoke_20260602T154538Z.json")
PREVIOUS_IMPROVED_PATH = Path("/private/tmp/saynow_3mvp_current_scenario_quality_improved_smoke_20260602T160405Z.json")
DEFAULT_BASE_URL = "http://43.202.146.182:8080"

USER_UTTERANCES = {
    1: "I like pizza because it is spicy.",
    2: "I cook sometimes but I am not good in cook.",
    3: "I ate tteokbokki yesterday with my friend.",
    4: "I want try sushi next because I never eat it before.",
    5: "I went to Busan last weekend.",
    6: "Most memorable part was see the sea at night.",
    7: "I went with my college friends.",
    8: "I would like to travel to Vancouver next.",
    9: "In morning I usually drinking water and check schedule.",
    10: "I spend free time to read books.",
    11: "I enjoy evening because I can relaxing after work.",
    12: "I want to change my sleeping habit because I sleep too late.",
}

EXPECTED_FEEDBACK_TYPES = {
    1: "GOOD",
    2: "NEEDS_IMPROVEMENT",
    3: "GOOD",
    4: "NEEDS_IMPROVEMENT",
    5: "GOOD",
    6: "NEEDS_IMPROVEMENT",
    7: "GOOD",
    8: "GOOD",
    9: "NEEDS_IMPROVEMENT",
    10: "NEEDS_IMPROVEMENT",
    11: "NEEDS_IMPROVEMENT",
    12: "GOOD",
}

EXPECTED_REASONS = {
    1: "좋아하는 음식과 이유가 분명한 답변.",
    2: "good in cook 구조가 어색한 답변.",
    3: "최근 먹은 음식, 시점, 동행이 분명한 답변.",
    4: "want to, have never eaten 같은 형태 보정이 필요한 답변.",
    5: "여행지와 시점이 분명한 답변.",
    6: "관사와 동명사 형태 보정이 필요한 답변.",
    7: "동행 대상이 분명한 답변.",
    8: "다음 여행 희망지가 분명한 답변.",
    9: "관사, 동사 형태, 소유격 보정이 필요한 답변.",
    10: "spend time 뒤 동명사 구조 보정이 필요한 답변.",
    11: "can 뒤 원형 동사가 필요한 답변.",
    12: "바꾸고 싶은 루틴과 이유가 분명한 답변.",
}


def main() -> int:
    base_url = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL).rstrip("/")
    generated_at = datetime.now(timezone.utc)
    output_path = Path(
        f"/private/tmp/saynow_3mvp_current_scenario_latest_smoke_{generated_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    scenarios_by_category = _load_scenarios(ATTACHMENT_PATH)
    health = _request_json("GET", f"{base_url}/health", None)
    schema_fields = _turn_feedback_schema_fields(base_url)
    result: dict[str, Any] = {
        "metadata": {
            "executedAt": generated_at.isoformat(),
            "attachmentPath": str(ATTACHMENT_PATH),
            "previousBaselinePath": str(PREVIOUS_BASELINE_PATH),
            "previousImprovedPath": str(PREVIOUS_IMPROVED_PATH),
            "outputPath": str(output_path),
            "targetBaseUrl": base_url,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "--short", "HEAD"),
            "health": health,
            "turnFeedbackSchemaFields": schema_fields,
        },
        "scenarios": [],
        "qualityIssues": [],
        "comparison": {},
    }

    scenario_index = 0
    for category in scenarios_by_category:
        for scenario in category["scenarios"]:
            scenario_index += 1
            scenario_result = _run_scenario(base_url, category, scenario, scenario_index)
            result["scenarios"].append(scenario_result)
            result["qualityIssues"].extend(scenario_result["qualityIssues"])

    result["comparison"] = _compare_with_previous(result)
    result["metadata"]["qualityIssueCount"] = len(result["qualityIssues"])
    result["metadata"]["finishedAt"] = datetime.now(timezone.utc).isoformat()

    output_path.write_text(json.dumps(_drop_none(result), ensure_ascii=False, indent=2) + "\n")
    print(output_path)
    print(json.dumps({
        "qualityIssueCount": result["metadata"]["qualityIssueCount"],
        "scenarioCount": len(result["scenarios"]),
        "outputPath": str(output_path),
    }, ensure_ascii=False))
    return 0


def _run_scenario(base_url: str, category: dict[str, Any], scenario: dict[str, Any], scenario_index: int) -> dict[str, Any]:
    session_id = 9800 + scenario_index
    scenario_context = {
        "scenarioId": scenario["scenarioId"],
        "title": scenario["title"],
        "briefing": scenario["briefing"],
        "conversationGoal": scenario["conversationGoal"],
    }
    turns = []
    next_questions = []
    quality_issues = []
    presented_ai_question = scenario["questions"][0]["questionEn"]
    presented_translated_question = scenario["questions"][0]["questionKo"]

    for question in scenario["questions"]:
        turn_id = scenario["scenarioId"] * 1000 + question["sequence"]
        user_utterance = USER_UTTERANCES[question["questionId"]]
        turn_request = {
            "sessionId": session_id,
            "turnId": turn_id,
            "sequence": question["sequence"],
            "scenario": scenario_context,
            "turn": {
                "aiQuestion": presented_ai_question,
                "translatedQuestion": presented_translated_question,
                "userUtterance": user_utterance,
            },
        }
        turn_started = time.perf_counter()
        turn_feedback_creation = _request_json("POST", f"{base_url}/api/v1/conversation/turn-feedback", turn_request)
        turn_elapsed_ms = round((time.perf_counter() - turn_started) * 1000, 2)
        turns.append({
            "questionId": question["questionId"],
            "sequence": question["sequence"],
            "turnId": turn_id,
            "fixedQuestionEn": question["questionEn"],
            "fixedQuestionKo": question["questionKo"],
            "presentedAiQuestion": presented_ai_question,
            "presentedTranslatedQuestion": presented_translated_question,
            "userUtterance": user_utterance,
            "expectedFeedbackType": EXPECTED_FEEDBACK_TYPES[question["questionId"]],
            "expectedReason": EXPECTED_REASONS[question["questionId"]],
            "turnFeedbackCreation": turn_feedback_creation,
            "elapsedMs": turn_elapsed_ms,
            "qualityIssues": [],
        })

        next_question = _find_next_question(scenario["questions"], question["sequence"])
        if next_question is None:
            continue

        next_request = {
            "sessionId": session_id,
            "submittedTurnId": turn_id,
            "submittedSequence": question["sequence"],
            "scenario": scenario_context,
            "currentTurn": {
                "aiQuestion": presented_ai_question,
                "translatedQuestion": presented_translated_question,
                "userUtterance": user_utterance,
            },
            "nextQuestion": {
                "questionId": next_question["questionId"],
                "sequence": next_question["sequence"],
                "questionEn": next_question["questionEn"],
                "questionKo": next_question["questionKo"],
            },
        }
        next_started = time.perf_counter()
        next_response = _request_json("POST", f"{base_url}/api/v1/conversation/next-question", next_request)
        next_elapsed_ms = round((time.perf_counter() - next_started) * 1000, 2)
        next_quality_issues = _evaluate_next_question(next_response, next_question)
        next_questions.append({
            "afterTurnId": turn_id,
            "nextQuestionId": next_question["questionId"],
            "nextSequence": next_question["sequence"],
            "fixedQuestionEn": next_question["questionEn"],
            "fixedQuestionKo": next_question["questionKo"],
            "response": next_response,
            "elapsedMs": next_elapsed_ms,
            "qualityIssues": next_quality_issues,
        })
        quality_issues.extend(
            f"scenario={scenario['scenarioId']} afterTurnId={turn_id}: {issue}"
            for issue in next_quality_issues
        )
        presented_ai_question = next_response["aiQuestion"]
        presented_translated_question = next_response["translatedQuestion"]

    session_started = time.perf_counter()
    session_feedback = _request_json("POST", f"{base_url}/api/v1/conversation/session-feedback", {
        "sessionId": session_id,
        "scenario": scenario_context,
        "expectedTurnIds": [turn["turnId"] for turn in turns],
    })
    session_elapsed_ms = round((time.perf_counter() - session_started) * 1000, 2)
    feedback_by_turn_id = {
        feedback["turnId"]: feedback
        for feedback in session_feedback.get("turnFeedbacks", [])
    }

    for turn in turns:
        feedback = feedback_by_turn_id.get(turn["turnId"])
        turn["turnFeedback"] = feedback
        turn_issues = _evaluate_turn_feedback(turn, feedback)
        turn["qualityIssues"] = turn_issues
        quality_issues.extend(
            f"scenario={scenario['scenarioId']} turnId={turn['turnId']}: {issue}"
            for issue in turn_issues
        )

    expected_counts = _count_expected_feedbacks(turns)
    actual_counts = _count_actual_feedbacks(session_feedback)
    session_issues = _evaluate_session_feedback(session_feedback, actual_counts)
    quality_issues.extend(
        f"scenario={scenario['scenarioId']} sessionId={session_id}: {issue}"
        for issue in session_issues
    )

    return {
        "categoryId": category["categoryId"],
        "categoryName": category["categoryName"],
        "scenarioId": scenario["scenarioId"],
        "sessionId": session_id,
        "title": scenario["title"],
        "briefing": scenario["briefing"],
        "conversationGoal": scenario["conversationGoal"],
        "turns": turns,
        "nextQuestions": next_questions,
        "sessionFeedback": session_feedback,
        "qualityIssues": quality_issues,
        "sessionFeedbackElapsedMs": session_elapsed_ms,
        "actualCounts": actual_counts,
        "expectedCounts": expected_counts,
    }


def _request_json(method: str, url: str, payload: dict[str, Any] | None) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with {exc.code}: {raw}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _turn_feedback_schema_fields(base_url: str) -> list[str]:
    openapi = _request_json("GET", f"{base_url}/openapi.json", None)
    schema = openapi["components"]["schemas"]["TurnFeedbackData"]
    return sorted(schema["properties"].keys())


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    end = text.rfind("]")
    if end == -1:
        raise ValueError("scenario attachment does not contain a JSON array")
    return json.loads(text[: end + 1])


def _find_next_question(questions: list[dict[str, Any]], sequence: int) -> dict[str, Any] | None:
    return next((question for question in questions if question["sequence"] == sequence + 1), None)


def _evaluate_next_question(response: dict[str, Any], next_question: dict[str, Any]) -> list[str]:
    issues = []
    ai_question = response.get("aiQuestion", "")
    translated_question = response.get("translatedQuestion", "")
    if next_question["questionEn"] not in ai_question:
        issues.append("aiQuestion에 고정 다음 질문 English가 포함되지 않음")
    if next_question["questionKo"].rstrip("?") not in translated_question:
        issues.append("translatedQuestion에 고정 다음 질문 Korean이 포함되지 않음")
    if ai_question.lower().startswith("i see."):
        issues.append("맞장구가 I see.로 시작함")
    return issues


def _evaluate_turn_feedback(turn: dict[str, Any], feedback: dict[str, Any] | None) -> list[str]:
    if feedback is None:
        return ["sessionFeedback.turnFeedbacks에 해당 turnId가 없음"]
    issues = []
    actual_type = feedback.get("feedbackType")
    expected_type = turn["expectedFeedbackType"]
    if actual_type != expected_type:
        issues.append(f"expectedFeedbackType={expected_type} 실제={actual_type}")
    better_expression = feedback.get("betterExpression")
    if actual_type == "NEEDS_IMPROVEMENT" and not better_expression:
        issues.append("NEEDS_IMPROVEMENT인데 betterExpression이 없음")
    if actual_type == "GOOD" and better_expression:
        issues.append("GOOD인데 betterExpression이 있음")
    for field in ["koreanAnalogy", "feedbackDetail"]:
        if not feedback.get(field):
            issues.append(f"{field}가 비어 있음")
    return issues


def _evaluate_session_feedback(session_feedback: dict[str, Any], actual_counts: dict[str, int]) -> list[str]:
    issues = []
    band = _score_band(actual_counts)
    if band is None:
        return ["turnFeedbacks가 비어 있어 점수 밴드를 계산할 수 없음"]
    min_score, max_score, label = band
    score = session_feedback.get("nativeScore")
    if score is None or not (min_score <= score <= max_score):
        issues.append(f"nativeScore={score}가 GOOD 비율 기준 범위 {min_score}-{max_score} 밖임")
    if session_feedback.get("nativeLevelLabel") != label:
        issues.append(f"nativeLevelLabel={session_feedback.get('nativeLevelLabel')}가 서버 기준 라벨 {label}와 다름")
    if not session_feedback.get("summary"):
        issues.append("summary가 비어 있음")
    return issues


def _score_band(counts: dict[str, int]) -> tuple[int, int, str] | None:
    total = counts["GOOD"] + counts["NEEDS_IMPROVEMENT"]
    if total == 0:
        return None
    good_ratio = counts["GOOD"] * 100 / total
    if good_ratio >= 90:
        return 90, 95, "원어민에 가까운 자연스러움"
    if good_ratio >= 75:
        return 82, 89, "유학생 느낌"
    if good_ratio >= 50:
        return 70, 81, "기초 회화 연습 단계"
    if good_ratio >= 25:
        return 60, 69, "문장 뼈대 연습 단계"
    return 50, 59, "기초 문장 교정 단계"


def _count_expected_feedbacks(turns: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"GOOD": 0, "NEEDS_IMPROVEMENT": 0}
    for turn in turns:
        counts[turn["expectedFeedbackType"]] += 1
    return counts


def _count_actual_feedbacks(session_feedback: dict[str, Any]) -> dict[str, int]:
    counts = {"GOOD": 0, "NEEDS_IMPROVEMENT": 0}
    for feedback in session_feedback.get("turnFeedbacks", []):
        feedback_type = feedback.get("feedbackType")
        if feedback_type in counts:
            counts[feedback_type] += 1
    return counts


def _compare_with_previous(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": _compare_with_path(result, PREVIOUS_BASELINE_PATH),
        "improved": _compare_with_path(result, PREVIOUS_IMPROVED_PATH),
    }


def _compare_with_path(result: dict[str, Any], path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    previous = json.loads(path.read_text())
    previous_by_scenario = {
        item["scenarioId"]: item
        for item in previous.get("scenarios", [])
    }
    rows = []
    for current in result["scenarios"]:
        previous_scenario = previous_by_scenario.get(current["scenarioId"])
        if previous_scenario is None:
            continue
        previous_feedback = previous_scenario.get("sessionFeedback", {})
        current_feedback = current.get("sessionFeedback", {})
        rows.append({
            "scenarioId": current["scenarioId"],
            "title": current["title"],
            "previousSessionId": previous_scenario.get("sessionId"),
            "currentSessionId": current.get("sessionId"),
            "previousCounts": previous_scenario.get("actualCounts"),
            "currentCounts": current.get("actualCounts"),
            "previousNativeScore": previous_feedback.get("nativeScore"),
            "currentNativeScore": current_feedback.get("nativeScore"),
            "previousNativeLevelLabel": previous_feedback.get("nativeLevelLabel"),
            "currentNativeLevelLabel": current_feedback.get("nativeLevelLabel"),
            "previousQualityIssueCount": len(previous_scenario.get("qualityIssues", [])),
            "currentQualityIssueCount": len(current.get("qualityIssues", [])),
        })
    return {
        "path": str(path),
        "metadata": previous.get("metadata", {}),
        "rows": rows,
    }


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
