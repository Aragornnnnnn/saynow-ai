# SayNow AI Server — 3차 MVP

## Project Overview

SayNow 백엔드가 호출하는 내부 AI 서버입니다. 3차 MVP에서는 슬롯 기반 역할극을 줄이고, 주제가 있는 프리톡을 4개 고정 질문으로 진행합니다. AI 서버는 질문 순서나 세션 상태를 결정하지 않고, 백엔드가 넘긴 고정 질문과 사용자 발화를 바탕으로 사용자에게 보일 AI 문장과 피드백을 생성합니다.

## Tech Stack

- **Framework**: FastAPI.
- **LLM**: OpenAI Chat Completions 또는 OpenAI 호환 Chat Completions.
- **Validation**: Pydantic v2.
- **Runtime State**: 턴별 피드백을 최종 피드백 호출 전까지 프로세스 메모리 캐시에 보관합니다.

## API Responsibilities

### `POST /api/v1/conversation/next-question`

- 백엔드가 `nextQuestion.questionEn`과 `nextQuestion.questionKo`로 다음 고정 질문을 전달합니다.
- AI 서버는 직전 사용자 발화에 짧게 반응하고 다음 고정 질문을 자연스럽게 이어 붙입니다.
- AI 서버는 다음 질문을 새로 고르거나 질문 의도를 바꾸지 않습니다.
- 응답은 `aiQuestion`, `translatedQuestion`만 반환합니다.

### `POST /api/v1/conversation/turn-feedback`

- 사용자 발화 1개를 `GOOD` 또는 `NEEDS_IMPROVEMENT`로 판단합니다.
- 모든 피드백은 `koreanAnalogy`를 포함합니다.
- `NEEDS_IMPROVEMENT`는 `correctionPoint`, `correctionReason`, `plusOneExpression`을 포함합니다.
- `GOOD`은 `praiseSummary`, `praiseReason`을 포함합니다.
- 생성된 턴별 피드백은 AI 서버 캐시에 저장하고 응답은 `PREPARING`을 반환합니다.

### `POST /api/v1/conversation/session-feedback`

- `expectedTurnIds`에 해당하는 캐시된 턴별 피드백을 조회합니다.
- 누락된 턴이 있으면 HTTP 409와 `TURN_FEEDBACK_NOT_READY`를 반환합니다.
- 최종 응답은 `nativeScore`, `nativeLevelLabel`, `summary`, `turnFeedbacks`를 포함합니다.

### `POST /api/v1/conversation/guide`

- 영어 학습 관련 질문만 처리합니다.
- 가이드 대화는 턴별 피드백과 최종 피드백 입력에 포함하지 않습니다.

## Quality Rules

- 이번 MVP의 최우선 목표는 응답 속도나 토큰 절감이 아니라 품질입니다.
- 피드백은 사용자의 실제 발화에 근거해야 합니다.
- 문법뿐 아니라 뉘앙스, 공손함, 상황 적절성, 어휘 선택, 질문에 대한 답변 적절성을 함께 봅니다.
- 잘한 발화는 억지로 고치지 않고 왜 좋은 발화였는지 설명합니다.
- JSON 응답 API에서는 JSON 외 설명 문장을 반환하지 않습니다.

## Verification

```bash
OPENAI_API_KEY=test-key python -m unittest discover -s tests -p 'test*.py'
python -m compileall app tests
git diff --check
```
