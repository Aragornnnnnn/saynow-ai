# SayNow AI Server

3차 MVP 백엔드가 호출하는 내부 AI 서버입니다.

## 역할

- 직전 사용자 발화에 대한 짧은 맞장구와 백엔드가 전달한 다음 고정 질문을 하나의 `aiQuestion`으로 연결합니다.
- 사용자 발화 1개에 대한 턴별 피드백을 생성하고 AI 서버 프로세스 메모리 캐시에 보관합니다.
- 최종 피드백 생성 시 캐시된 턴별 피드백을 모아 `nativeScore`, `nativeLevelLabel`, `summary`와 함께 반환합니다.
- 영어 학습 가이드 질문은 기존 `guide` API로 계속 처리합니다.

슬롯 완료 판정, 세션/턴 생성, DB 저장, NPS, 최종 완료 상태 관리는 백엔드 책임입니다.

## API

### `POST /api/v1/conversation/next-question`

백엔드가 다음 고정 질문을 `nextQuestion`으로 전달하면 AI 서버는 직전 발화에 대한 짧은 반응과 해당 고정 질문을 자연스럽게 이어 붙입니다. AI 서버는 다음 질문을 새로 고르지 않습니다.

응답.

```json
{
  "aiQuestion": "Oh, you like spicy pizza. Do you cook often?",
  "translatedQuestion": "매운 피자를 좋아하는군요. 요리는 자주 하나요?"
}
```

### `POST /api/v1/conversation/turn-feedback`

사용자 발화 1개에 대한 피드백을 생성하고 AI 서버 캐시에 저장합니다. 백엔드는 이 응답을 받더라도 턴별 피드백을 즉시 DB에 저장하지 않고, 최종 피드백 생성 시 한 번에 저장합니다.

응답.

```json
{
  "sessionId": 1000,
  "turnId": 5000,
  "feedbackStatus": "PREPARING"
}
```

### `POST /api/v1/conversation/session-feedback`

`expectedTurnIds`에 해당하는 턴별 피드백을 캐시에서 조회한 뒤 세션 최종 피드백을 생성합니다. 필요한 턴 피드백이 없으면 HTTP 409와 `TURN_FEEDBACK_NOT_READY`를 반환합니다.

응답.

```json
{
  "sessionId": 1000,
  "nativeScore": 82,
  "nativeLevelLabel": "유학생 수준",
  "summary": "하고 싶은 말을 끝까지 전달하는 힘이 좋았어요. 간접의문문 어순만 조금 다듬으면 더 자연스러워요.",
  "turnFeedbacks": [
    {
      "turnId": 5000,
      "feedbackType": "GOOD",
      "koreanAnalogy": "한국어로 비유하자면 '저는 피자가 좋아요. 매워서요'처럼 담백하게 들려요.",
      "correctionPoint": null,
      "correctionReason": null,
      "plusOneExpression": null,
      "praiseSummary": "이유를 because로 자연스럽게 붙였어요.",
      "praiseReason": "좋아하는 음식과 이유를 한 문장 안에서 분명하게 연결했어요."
    }
  ]
}
```

### `POST /api/v1/conversation/guide`

시나리오 대화 중 영어 표현, 문법, 단어, 뉘앙스를 질문할 때 사용합니다. 가이드 대화는 턴별 피드백이나 최종 피드백 입력에 포함하지 않습니다.

## 피드백 기준

3차 MVP의 최우선 목표는 응답 속도나 토큰 절감이 아니라 품질입니다. 턴별 피드백은 문법만 보지 않고 뉘앙스, 공손함, 상황 적절성, 어휘 선택, 질문에 대한 답변 적절성을 함께 판단합니다.

`NEEDS_IMPROVEMENT`에는 `koreanAnalogy`, `correctionPoint`, `correctionReason`, `plusOneExpression`을 반드시 포함합니다. `GOOD`에는 `koreanAnalogy`, `praiseSummary`, `praiseReason`을 반드시 포함합니다.

## Error Policy

- 잘못된 요청은 HTTP 400과 `{"code": "INVALID_REQUEST", "message": "잘못된 요청입니다."}`를 반환합니다.
- 필요한 턴별 피드백이 아직 없으면 HTTP 409와 `TURN_FEEDBACK_NOT_READY`를 반환합니다.
- LLM 호출 실패나 계약에 맞지 않는 LLM 응답은 HTTP 500과 `AI_GENERATION_FAILED`를 반환합니다.

## Environment Variables

```bash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
SENTRY_DSN=
```

## Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Verification

```bash
OPENAI_API_KEY=test-key python -m unittest discover -s tests -p 'test*.py'
python -m compileall app tests
git diff --check
```
