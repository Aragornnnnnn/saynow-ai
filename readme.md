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
  "userUtterance": "I want iced americano.",
  "scenarioTitle": "카페에서 주문하기",
  "scenarioSituation": "사용자는 카페 직원과 대화하며 테이크아웃 음료를 주문해야 한다.",
  "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
  "slots": [
    {
      "slotName": "drink",
      "filled": false
    },
    {
      "slotName": "size",
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
  "filledSlots": [
    {
      "slotName": "drink"
    }
  ],
  "turnClassification": "ANSWER"
}
```

`filledSlots`는 이번 발화로 새롭게 충족된 슬롯만 포함합니다. 남은 미충족 슬롯이 없으면 `nextQuestion`과 `translatedQuestion`은 `null`입니다.

### `POST /api/v1/conversation/feedback`

요청.

```json
{
  "scenarioTitle": "카페에서 주문하기",
  "scenarioSituation": "사용자는 카페 직원과 대화하며 테이크아웃 음료를 주문해야 한다.",
  "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
  "sessionResult": "SUCCESS",
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

요청은 `POST /api/v1/conversation/feedback`과 같습니다. `scenarioSituation`은 SSE 피드백 요약과 턴별 피드백 생성에도 동일하게 사용합니다.

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

## Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
