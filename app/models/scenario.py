# 시나리오 Pydantic 스키마 — 시나리오 데이터 구조 및 API 응답 형식 정의
from pydantic import BaseModel


class Scenario(BaseModel):
    id: str
    category: str # ex: cafe
    title: str # ex: "카페에서 주문하기"
    situation: str # ex: "당신은 카페에 들어가서 음료를 주문하려고 합니다. 바리스타와 대화를 나누며 원하는 음료를 주문하세요."
    goal: str # ex: "카페에서 원하는 음료를 성공적으로 주문하기"
    required_info: list[str] # ex: ["음료 종류", "사이즈", "추가 옵션"] 클리어 판단 기준 (유저한테는 안 보임)
    max_questions: int = 5 # 최대 꼬리 질문 수 

    @property
    def max_follow_up_count(self) -> int:
        return self.max_questions


class ScenarioListResponse(BaseModel):
    success: bool  # API 요청 성공 여부
    data: list[Scenario]  # Scenario가 여러 개 담긴 리스트
    error: str | None = None


class ScenarioDetailResponse(BaseModel):
    success: bool
    data: Scenario | None = None  # Scenario 하나, 없으면 null
    error: str | None = None
