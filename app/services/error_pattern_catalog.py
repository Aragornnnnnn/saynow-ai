# 한국인 영어 오류 패턴 seed 데이터를 로드하고 프롬프트용 catalog를 만든다.
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any


_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "error_patterns.json"
_DETECTED_PATTERN_STATUSES = {"correct", "incorrect", "attempted"}


@dataclass(frozen=True)
class ErrorPattern:
    error_type: str
    display_name: str
    korean_pct: float | None
    breaks_meaning: bool
    correction_priority: str
    gamifiable: bool
    level_tone: str
    feedback_copy: str
    example_wrong: str
    example_right: str
    denominator_type: str
    source: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorPattern":
        return cls(
            error_type=str(data["error_type"]),
            display_name=str(data["display_name"]),
            korean_pct=data.get("korean_pct"),
            breaks_meaning=bool(data["breaks_meaning"]),
            correction_priority=str(data["correction_priority"]),
            gamifiable=bool(data["gamifiable"]),
            level_tone=str(data["level_tone"]),
            feedback_copy=str(data["feedback_copy"]),
            example_wrong=str(data["example_wrong"]),
            example_right=str(data["example_right"]),
            denominator_type=str(data["denominator_type"]),
            source=str(data["source"]),
        )


@dataclass(frozen=True)
class DetectedErrorPattern:
    error_type: str
    status: str
    evidence: str
    pattern: ErrorPattern

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "errorType": self.error_type,
            "status": self.status,
            "evidence": self.evidence,
            "koreanPct": self.pattern.korean_pct,
            "breaksMeaning": self.pattern.breaks_meaning,
            "correctionPriority": self.pattern.correction_priority,
            "gamifiable": self.pattern.gamifiable,
            "feedbackCopy": self.pattern.feedback_copy,
        }


@lru_cache(maxsize=1)
def load_error_patterns() -> dict[str, ErrorPattern]:
    raw_patterns = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    patterns = [ErrorPattern.from_dict(raw_pattern) for raw_pattern in raw_patterns]
    return {pattern.error_type: pattern for pattern in patterns}


def get_error_pattern(error_type: str) -> ErrorPattern | None:
    return load_error_patterns().get(error_type)


def prompt_error_pattern_catalog() -> str:
    lines = []
    for pattern in load_error_patterns().values():
        pct = "unknown" if pattern.korean_pct is None else _format_pct(pattern.korean_pct)
        breaks_meaning = "true" if pattern.breaks_meaning else "false"
        gamifiable = "true" if pattern.gamifiable else "false"
        lines.append(
            f"- {pattern.error_type}: {pattern.display_name}, korean_pct={pct}, "
            f"breaks_meaning={breaks_meaning}, priority={pattern.correction_priority}, "
            f"gamifiable={gamifiable}, copy={pattern.feedback_copy}"
        )
    return "\n".join(lines)


def parse_detected_patterns(raw_value: Any) -> tuple[DetectedErrorPattern, ...]:
    if not isinstance(raw_value, list):
        return ()
    detected_patterns: list[DetectedErrorPattern] = []
    for item in raw_value:
        if not isinstance(item, dict):
            continue
        error_type = str(item.get("errorType") or item.get("error_type") or "").strip()
        status = str(item.get("status") or "").strip()
        pattern = get_error_pattern(error_type)
        if pattern is None or status not in _DETECTED_PATTERN_STATUSES:
            continue
        detected_patterns.append(
            DetectedErrorPattern(
                error_type=error_type,
                status=status,
                evidence=str(item.get("evidence") or "").strip(),
                pattern=pattern,
            )
        )
    return tuple(detected_patterns)


def _format_pct(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)
