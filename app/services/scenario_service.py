# 시나리오 서비스 — scenarios.json에서 시나리오 데이터를 읽어 조회하는 로직
import json
from pathlib import Path
from app.models.scenario import Scenario

_DATA_PATH = Path(__file__).parent.parent / "data" / "scenarios.json"


def _load() -> list[Scenario]:
    with open(_DATA_PATH, encoding="utf-8") as f:
        return [Scenario(**item) for item in json.load(f)]


def get_all() -> list[Scenario]:
    return _load()


def get_by_id(scenario_id: str) -> Scenario | None:
    return next((s for s in _load() if s.id == scenario_id), None)
