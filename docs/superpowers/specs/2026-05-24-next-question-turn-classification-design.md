# Next Question Turn Classification Design

## Goal

`filledSlots=[]`가 의미하는 상태를 분리한다. 슬롯이 채워지지 않았더라도 사용자가 추천이나 추가 정보를 요청했다면 AI 서버는 그 발화를 정상 대화 진행으로 분류하고, 백엔드는 이 분류를 기준으로 하트 차감 여부를 결정할 수 있어야 한다.

## Current Problem

현재 `POST /api/v1/conversation/next-question` 응답은 `filledSlots`, `nextQuestion`, `translatedQuestion`만 내려준다. 이 구조에서는 아래 두 케이스가 모두 `filledSlots=[]`로 보인다.

- `Can I see the menu?`처럼 사용자가 추가 정보를 얻기 위해 정상 질문을 한 경우.
- `I want drink.`, `I want.`, nonsense처럼 슬롯을 채울 수 없는 실패 발화.

백엔드가 `filledSlots=[]`만 보고 하트를 차감하면 정상 질문도 실패로 처리된다.

또 사용자는 AI 응답 텍스트만 볼 수 있다. 따라서 메뉴 요청에 `Here are the menu options`처럼 실제 메뉴가 빠진 응답을 주면 사용자는 선택할 근거를 얻지 못한다.

## Decision

AI 서버 응답에 `turnClassification`을 추가한다. AI 서버는 발화의 언어적 성격만 분류하고, 하트 차감 정책은 백엔드가 결정한다. AI 서버 응답에는 `validProgress`나 `shouldDeductHeart` 같은 정책 boolean을 넣지 않는다.

```json
{
  "filledSlots": [],
  "nextQuestion": "The menu includes iced Americano, latte, cappuccino, and tea. What would you like to order?",
  "translatedQuestion": "메뉴에는 아이스 아메리카노, 라떼, 카푸치노, 차가 있어요. 무엇을 주문하시겠어요?",
  "turnClassification": "ASSISTANCE_REQUEST"
}
```

## Classification Values

| 값 | 의미 | 백엔드 하트 정책 권장 |
| --- | --- | --- |
| `ANSWER` | 사용자가 현재 AI 질문에 답했다. 슬롯 답변, 선택형 답변, 옵션 완료 답변을 포함한다. | 차감 없음 |
| `ASSISTANCE_REQUEST` | 사용자가 추천, 메뉴, 옵션, 가능 선택지, 규칙, 세부 정보 같은 도움을 요청했다. | 차감 없음 |
| `INVALID_RESPONSE` | 사용자가 질문에 답하지 못했거나, off-topic, nonsense, generic object, incomplete fragment를 말했다. | 차감 |

## Why Three States

초기 설계는 `SLOT_ANSWER`, `RECOMMENDATION_REQUEST`, `INFORMATION_REQUEST`, `OPTION_COMPLETION`, `INVALID_RESPONSE` 5상태였다. 하지만 하트 차감 정책 관점에서는 추천 요청과 정보 요청이 모두 정상적인 도움 요청이고, 옵션 완료는 별도 상태가 아니라 현재 질문에 대한 자연스러운 답변이다.

따라서 AI 서버 계약은 모델이 구분해야 하는 핵심 판단만 남긴다. 사용자가 답했는지, 도움을 요청했는지, 실패 발화인지가 백엔드 정책에 필요한 최소 분류다.

## AI Server Behavior

- `filledSlots`는 기존처럼 이번 발화로 새롭게 채워진 슬롯만 담는다.
- `turnClassification`은 `filledSlots`와 독립적으로 항상 내려준다.
- 슬롯이 하나 이상 채워지면 기본 분류는 `ANSWER`다.
- `That’s all.`, `No sugar, please.`, `Oat milk and no sugar, please.`처럼 옵션 질문을 끝내거나 선호를 명확히 답하면 `ANSWER`다.
- `Can you recommend a menu?`, `What do you recommend?`, `Can I see the menu?`는 `ASSISTANCE_REQUEST`다.
- `I want.`, `I want drink.`, `My shoes are swimming in the moon today.`, `I don't know.`는 `INVALID_RESPONSE`다.
- 메뉴, 옵션, 선택지, 규칙, 세부 정보를 묻는 경우 `nextQuestion`에는 사용자가 실제로 선택하거나 답변할 수 있는 구체 정보를 포함한다.

## Backend Integration

백엔드는 `turnClassification`을 하트 정책의 source of truth로 사용한다. `filledSlots=[]` 자체는 하트 차감 사유가 아니다. 백엔드는 `INVALID_RESPONSE`일 때만 하트를 차감하고, 나머지 분류는 정상 대화 진행으로 처리한다.

프론트에는 백엔드가 계산한 `heartDeducted`, `remainingHearts` 같은 결과 필드를 내려주는 것이 좋다. 이 필드는 AI 서버 계약에 넣지 않는다.

## Compatibility

이 변경은 `next-question` 응답 JSON에 필드를 추가하는 방식이다. 백엔드가 새 필드를 읽도록 변경되어야 하며, 백엔드 반영 전에는 기존 하트 차감 문제가 완전히 해결되지 않는다.

AI 서버는 모델이 이전 5상태 값을 반환하더라도 내부에서 새 3상태 값으로 정규화한다.

## Test Strategy

- 서비스 테스트에서 추천 요청, 정보 요청, 옵션 완료, 슬롯 답변, 실패 발화가 각각 기대 분류를 반환하는지 확인한다.
- 메뉴 요청에서 모델이 실제 메뉴를 빠뜨리면 AI 서버가 사용자가 볼 수 있는 메뉴 항목을 `nextQuestion`에 보정하는지 확인한다.
- route 테스트에서 `turnClassification` 필드가 HTTP 응답에 포함되는지 확인한다.
- 기존 `filledSlots`, `nextQuestion`, `translatedQuestion` 동작은 유지한다.
- 전체 unittest와 compileall, diff check를 실행한다.
