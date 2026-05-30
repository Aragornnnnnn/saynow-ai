# SayNow AI Server

2차 MVP 백엔드가 호출하는 내부 AI 서버입니다.

## 역할

- 사용자 텍스트 발화를 바탕으로 새롭게 충족된 슬롯을 판단합니다.
- 미충족 슬롯이 남아 있으면 다음 영어 꼬리 질문과 한국어 번역을 생성합니다.
- 완료된 대화 세션의 전체 피드백과 턴별 피드백을 생성합니다.

STT, TTS, 세션 완료 판정, 누적 슬롯 저장은 백엔드 책임입니다.

## API

### `POST /api/v1/conversation/next-question`

요청.

```json
{
  "originalQuestion": "What would you like to order?",
  "originalQuestionTargetSlotName": "drink",
  "userUtterance": "I want iced americano.",
  "scenarioTitle": "카페에서 주문하기",
  "scenarioSituation": "사용자는 카페 직원과 대화하며 테이크아웃 음료를 주문해야 한다.",
  "aiRole": "카페 직원",
  "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
  "slots": [
    {
      "slotName": "drink",
      "description": "사용자가 주문하려는 음료 이름이나 종류를 구체적으로 말했는지 여부",
      "filled": false,
      "evidencePolicy": {
        "mode": "semantic_evidence",
        "hints": ["coffee", "latte", "tea"],
        "requiresEvidenceText": true,
        "mustBeGroundedIn": "latest_user_utterance"
      }
    },
    {
      "slotName": "size",
      "description": "사용자가 음료의 크기나 사이즈를 말했는지 여부",
      "filled": false
    }
  ]
}
```

응답.

```json
{
  "nextQuestion": "What size would you like?",
  "translatedQuestion": "어떤 사이즈로 드릴까요?",
  "nextQuestionTargetSlotName": "size",
  "filledSlots": [
    {
      "slotName": "drink"
    }
  ],
  "turnClassification": "ANSWER"
}
```

`originalQuestionTargetSlotName`은 직전 질문이 주로 겨냥한 슬롯입니다. 이 값은 `filledSlots`를 하나로 제한하지 않습니다. 최신 발화가 여러 슬롯의 근거를 담고 있으면 여러 슬롯을 함께 반환할 수 있습니다.
`filledSlots`는 이번 발화로 새롭게 충족된 슬롯만 포함합니다. 남은 미충족 슬롯이 없으면 `nextQuestion`, `translatedQuestion`, `nextQuestionTargetSlotName`은 `null`입니다.
`nextQuestionTargetSlotName`은 다음 질문이 주로 겨냥하는 슬롯이며, 백엔드는 이 값을 다음 요청의 `originalQuestionTargetSlotName`으로 전달합니다.
`slots[].evidencePolicy`는 문자열 JSON이 아니라 JSON object입니다. `semantic_evidence`는 정확한 키워드 일치가 아니라, 최신 사용자 발화에서 외국인이 슬롯 의도를 합리적으로 이해할 수 있는지 검증하는 방식입니다. `hints`는 대표 표현일 뿐 정답 단어 전체 목록이 아닙니다.

### `POST /api/v1/conversation/guide`

시나리오 대화 중 사용자가 영어 표현, 문법, 단어, 뉘앙스를 질문할 때 사용합니다. 가이드 모드 대화는 최종 피드백 입력에 포함하지 않습니다.

요청.

```json
{
  "question": "I would like coffee에서 would는 왜 쓰나요? I want coffee라고 하면 안 되나요?",
  "scenarioTitle": "카페에서 주문하기",
  "scenarioSituation": "사용자는 카페 직원과 대화하며 테이크아웃 음료를 주문해야 한다.",
  "aiRole": "카페 직원",
  "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다."
}
```

응답.

```json
{
  "answer": "would는 공손한 요청이나 가정 느낌을 줄 때 써요. 이 상황에서는 I'd like coffee가 I want coffee보다 부드럽게 들려요."
}
```

`question`은 영어 학습 관련 질문만 허용합니다. 프롬프트 인젝션, 시스템 지시 공개, 역할 변경, 코딩, 뉴스, 금융처럼 영어 학습 목적 밖의 요청은 모델을 호출하지 않고 안내 답변을 반환합니다.

### `POST /api/v1/conversation/feedback`

요청.

```json
{
  "scenarioTitle": "카페에서 주문하기",
  "scenarioSituation": "사용자는 카페 직원과 대화하며 테이크아웃 음료를 주문해야 한다.",
  "aiRole": "카페 직원",
  "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
  "sessionResult": "SUCCESS",
  "slots": [
    {
      "slotName": "drink",
      "description": "사용자가 주문하려는 음료 이름이나 종류를 구체적으로 말했는지 여부",
      "filled": true
    },
    {
      "slotName": "size",
      "description": "사용자가 음료의 크기나 사이즈를 말했는지 여부",
      "filled": true
    }
  ],
  "turns": [
    {
      "turnId": 101,
      "originalQuestion": "What would you like to order?",
      "userUtterance": "I want iced americano."
    }
  ]
}
```

응답.

```json
{
  "comprehensionScore": 82,
  "feedbackSummary": "전체적으로 의도는 잘 전달됐지만 주문 표현이 조금 짧게 들립니다.",
  "turnFeedbacks": [
    {
      "turnId": 101,
      "feedbackRequired": true,
      "nativeUnderstanding": "아이스 아메리카노를 주문하고 싶다는 의미로 이해됩니다.",
      "nativeLanguageInterpretation": "나 아이스 아메리카노 원해처럼 조금 직접적으로 들립니다.",
      "betterExpression": "I'd like an iced Americano, please."
    }
  ]
}
```

### `POST /api/v1/conversation/feedback/stream`

요청은 `POST /api/v1/conversation/feedback`과 같습니다. `scenarioSituation`, `aiRole`, `slots`는 SSE 피드백 요약과 턴별 피드백 생성에도 동일하게 사용합니다.

`scenarioSituation`은 사용자가 놓인 상황입니다. `aiRole`은 AI가 롤플레이에서 맡아야 하는 상대방 역할입니다. AI는 `aiRole`을 유지하며, 사용자에게 다른 사람에게 물어보라고 지시하지 않습니다. `scenarioGoal`은 사용자가 달성해야 하는 말하기 목표입니다.

`slots[].description`은 슬롯을 채웠다고 판단하는 의미 기준입니다. 특정 영어 표현을 강제하지 않고, 사용자가 어떤 의도를 전달하면 해당 슬롯을 채운 것으로 볼지 설명합니다.

응답은 `text/event-stream`입니다.

```text
event: summary
data: {"comprehensionScore":82,"feedbackSummary":"전체적으로 의도는 잘 전달됐지만 주문 표현이 조금 짧게 들립니다."}

event: turnFeedback
data: {"turnId":101,"feedbackRequired":true,"nativeUnderstanding":"외국인은 사용자가 아이스 아메리카노를 원한다고 이해했어요.","nativeLanguageInterpretation":"한국어로 비유하자면, '아이스 아메리카노 원해요'처럼 들려요.","betterExpression":"I'd like an iced Americano, please."}

event: done
data: {"turnCount":1}
```

## 피드백 판단 기준

`feedbackRequired`는 내부 점수표와 조건을 함께 적용해 판단합니다.

- `0-39`: 의도가 거의 전달되지 않거나 질문과 무관한 응답.
- `40-59`: 대략적인 의도만 보이고 핵심 정보가 빠지거나 크게 왜곡된 응답.
- `60-74`: 주된 의도는 이해되지만 문법, 단어 선택, 어순이 명확히 어색한 응답.
- `75-84`: 시나리오 의도는 분명하지만 자연스러움, 공손함, 완성도를 조금 다듬어야 하는 응답.
- `85-100`: 질문에 직접 답했고, 원어민이 추가 추측 없이 이해 가능하며, 의미를 막는 문법이나 단어 문제가 없는 응답.

`feedbackRequired=false`는 내부 점수가 `85-100`이고, 질문 의도에 답했으며, 해당 턴의 시나리오 의도를 충족하고, 추가 추측 없이 이해 가능하고, 의미를 막는 오류가 없을 때만 반환합니다. 그 외에는 `feedbackRequired=true`입니다.

`betterExpression`은 사용자 발화에서 딱 한 단계만 개선합니다. 원래 의도, 단어 수준, 문장 형태를 최대한 유지하고, 관사 하나, 공손한 표현 하나, 어순 하나처럼 가장 작은 개선만 적용합니다.

## Observability

Sentry DSN이 없으면 Sentry 초기화는 비활성화됩니다. DSN이 전달되면 아래 환경변수로 운영 오류 수집을 켤 수 있습니다.

```env
SENTRY_DSN=
SENTRY_ENVIRONMENT=develop
SENTRY_TRACES_SAMPLE_RATE=0.0
SENTRY_MAX_BREADCRUMBS=100
LOG_LEVEL=INFO
```

AI workflow는 주요 단계별 소요 시간을 `workflow`, `stage`, `duration_ms` 형태로 남깁니다. 현재 대상은 `next_question`, `feedback`, `feedback_summary`, `turn_feedback`, `feedback_review`, `feedback_repair`, `guide`입니다.

배포 workflow는 SSM Parameter Store의 `/saynow/develop` 또는 `/saynow/prod` 경로를 `.env`로 변환합니다. DSN은 코드에 넣지 말고 `SENTRY_DSN` 파라미터로 저장합니다. 일반 로그는 INFO 이상을 Sentry breadcrumb로 붙이고, 오류 이벤트는 API 라우터와 전역 500 handler의 `capture_exception` 경계에서 전송합니다.

## Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
