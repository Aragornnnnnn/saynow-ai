# Safety Guard And Guide Mode Design

## Goal

사용자가 시나리오 대화 중 영어 학습 질문을 할 수 있는 가이드 모드를 추가하고, 프롬프트 인젝션과 목적 외 사용 방어를 모든 LLM 입력 경계에 공통 적용한다.

## Scope

- `next-question`은 기존 응답 계약을 유지한다. 차단 대상 입력은 `INVALID_RESPONSE`로 처리하고 같은 목적의 질문을 다시 묻는다.
- `feedback`과 `feedback/stream`은 사용자 발화를 실행할 지시가 아니라 평가 대상 데이터로만 취급한다.
- `guide`는 영어 단어, 표현, 문법, 뉘앙스, 발음, 대체 표현 질문만 답한다.
- 가이드 대화는 피드백 요청 모델과 분리되어 최종 피드백에 포함되지 않는다.

## Architecture

공통 `safety_guard` 서비스가 사용자 입력을 검사한다. 서비스별 허용 범위는 목적 enum으로 나누고, 차단 응답은 API 계약 안에서 자연스럽게 처리한다.

`conversation_service`는 모델 호출 전 `next-question` 입력을 검사하고, 모든 system prompt에 공통 안전 정책을 포함한다. 새 가이드 서비스 함수는 `GuideChatRequest`를 받아 safety guard를 먼저 통과한 뒤, JSON 응답을 생성해 `GuideChatResponse`로 검증한다.

## API Contract

`POST /api/v1/conversation/guide`

Request fields.

- `question`: 사용자의 영어 학습 질문.
- `scenarioTitle`, `scenarioSituation`, `aiRole`, `scenarioGoal`: 현재 시나리오 컨텍스트.
- `originalQuestion`: 현재 또는 직전 AI 질문. 선택 필드.
- `userUtterance`: 사용자가 질문하게 된 직전 발화. 선택 필드.

Response fields.

- `answer`: 한국어 중심 답변. 필요할 때 짧은 영어 예시를 포함한다.

## Safety Policy

공통 차단 대상.

- 이전 지시, 시스템 프롬프트, 개발자 지시, 정책을 무시하거나 공개하라는 요청.
- 모델 역할을 영어 학습 도우미나 시나리오 상대역 밖으로 바꾸려는 요청.
- 코딩, 정치, 금융, 뉴스, 일반 검색처럼 영어 학습이나 현재 시나리오 진행과 무관한 요청.

## Testing

- 가이드 라우트가 문서화된 응답 모양을 반환한다.
- 영어 학습 질문은 LLM을 호출하고 답변을 반환한다.
- 프롬프트 인젝션과 목적 외 질문은 LLM을 호출하지 않고 안내 답변을 반환한다.
- `next-question` 프롬프트 인젝션은 `INVALID_RESPONSE`로 처리한다.
- 피드백 계열 프롬프트는 사용자 발화를 데이터로만 취급하는 공통 안전 정책을 포함한다.
