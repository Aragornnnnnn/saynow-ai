# 시나리오 라우터 — GET /scenarios (전체 목록), GET /scenarios/{id} (단건 조회)
from fastapi import APIRouter
from app.models.scenario import ScenarioListResponse, ScenarioDetailResponse
from app.services import scenario_service

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.get("", response_model=ScenarioListResponse, summary="전체 시나리오 조회")
def list_scenarios():
    """
    사용할 수 있는 모든 시나리오 목록을 반환합니다.

    5개 카테고리(공항, 호텔, 카페, 식당, 택시) × 2개 시나리오씩 총 10개
    """
    return ScenarioListResponse(success=True, data=scenario_service.get_all())


@router.get("/{scenario_id}", response_model=ScenarioDetailResponse, summary="시나리오 상세 조회")
def get_scenario(scenario_id: str):
    """
    특정 시나리오의 상세 정보를 조회합니다.

    - scenario_id: 시나리오 ID (예: cafe_1, airport_2)
    - 반환: 시나리오 제목, 상황, 목표, 필요 정보, 최대 질문 수
    """
    scenario = scenario_service.get_by_id(scenario_id)
    if not scenario:
        return ScenarioDetailResponse(success=False, error="Scenario not found")
    return ScenarioDetailResponse(success=True, data=scenario)
