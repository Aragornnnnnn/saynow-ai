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
  ]
}
```

`filledSlots`는 이번 발화로 새롭게 충족된 슬롯만 포함합니다. 남은 미충족 슬롯이 없으면 `nextQuestion`과 `translatedQuestion`은 `null`입니다.

### `POST /api/v1/conversation/feedback`

요청.

```json
{
  "scenarioTitle": "카페에서 주문하기",
  "scenarioGoal": "원하는 음료를 자연스럽게 주문할 수 있다.",
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

## Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
