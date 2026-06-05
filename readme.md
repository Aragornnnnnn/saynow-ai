# SayNow AI Server

3차 MVP 백엔드가 호출하는 내부 AI 서버입니다.

## 역할

- 직전 사용자 발화에 대한 짧은 맞장구와 백엔드가 전달한 다음 고정 질문을 하나의 `aiQuestion`으로 연결합니다.
- 사용자 발화 1개에 대한 턴별 피드백을 생성하고 AI 서버 프로세스 메모리 캐시에 최대 3시간 보관합니다.
- 최종 피드백 생성 시 캐시된 턴별 피드백을 모아 `nativeScore`, `highlightMessage`와 함께 반환합니다.
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

사용자 발화 1개에 대한 피드백을 생성하고 AI 서버 캐시에 저장합니다. 백엔드는 이 응답을 받더라도 턴별 피드백을 즉시 DB에 저장하지 않고, 최종 피드백 생성 시 한 번에 저장합니다. 캐시는 3시간 뒤 만료되고, 최종 피드백 생성이 성공하면 해당 세션 캐시를 삭제합니다.

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
  "nativeScore": 78,
  "highlightMessage": "한국인 40%가 헷갈리는 간접의문문 어순을 바로잡을 사람",
  "turnFeedbacks": [
    {
      "turnId": 5000,
      "feedbackType": "GOOD",
      "koreanAnalogy": "한국어로 비유하자면, \"저는 피자가 좋아요. 매워서요\"라고 자연스럽게 이유를 붙여 말하는 것과 같아요.",
      "feedbackDetail": "이유를 because로 자연스럽게 붙였고, 좋아하는 음식과 이유를 한 문장 안에서 분명하게 연결했어요.",
      "positiveFeedback": null,
      "benchmarkMessage": "한국인 79%가 놓치는 a/an 자리를 정확히 쓴 사람"
    },
    {
      "turnId": 5001,
      "feedbackType": "NEEDS_IMPROVEMENT",
      "koreanAnalogy": "한국어로 비유하자면, \"그게 뭔지 모르겠어\"라고 말하려다 어순이 살짝 꼬인 문장으로 말하는 것과 같아요.",
      "positiveFeedback": "어려운 간접의문문 구조에 도전한 점이 좋아요. 틀렸더라도 그 시도 자체가 다음 단계로 가는 재료예요.",
      "feedbackDetail": "what is it → what it is. 간접의문문에서는 의문문 어순이 아니라 평서문 어순을 써야 해요.",
      "benchmarkMessage": null
    }
  ]
}
```

### `POST /api/v1/conversation/guide`

시나리오 대화 중 영어 표현, 문법, 단어, 뉘앙스를 질문할 때 사용합니다. 가이드 대화는 턴별 피드백이나 최종 피드백 입력에 포함하지 않습니다.

## 피드백 기준

3차 MVP의 최우선 목표는 응답 속도나 토큰 절감이 아니라 품질입니다. 턴별 피드백은 문법만 보지 않고 뉘앙스, 공손함, 상황 적절성, 어휘 선택, 질문에 대한 답변 적절성을 함께 판단합니다.

`nativeScore`는 0-100 점수이며 100에 가까울수록 원어민 쪽에 가깝습니다. 세션 점수는 시도 단어수 20%, 문장 복잡도 30%, 이해 가능성 50%를 내부 합산해 계산합니다.

`highlightMessage`는 전체 총평이 아니라 발화별 피드백을 열어 보게 만드는 칭호형 후킹 문구입니다. 근거가 있으면 `한국인 40%가 헷갈리는 간접의문문 어순을 바로잡을 사람`처럼 정량 수치를 포함한 마침표 없는 명사구를 우선합니다.

`koreanAnalogy`는 문법 설명이 아니라 원래 영어가 한국어 감각으로 어떻게 들리는지 보여주는 필드입니다. `한국어로 비유하자면, "..."라고 ...하는 것과 같아요.` 형식을 우선합니다. raw JSON에서는 문자열 안 큰따옴표가 `\"`로 escape되지만, 클라이언트에서 JSON을 파싱해 렌더링하면 역슬래시는 보이지 않습니다.

`NEEDS_IMPROVEMENT`에는 `koreanAnalogy`, `positiveFeedback`, `feedbackDetail`을 반드시 포함합니다. `feedbackDetail`은 전체 발화를 반복하기보다 `what is it → what it is`처럼 가장 짧은 의미 단위의 before→after를 먼저 보여주고, 바로 뒤에 한국어 이유를 붙입니다. `benchmarkMessage`는 `null`로 둡니다. `GOOD`에는 `koreanAnalogy`, `feedbackDetail`을 반드시 포함하고, 근거가 있는 경우에만 `benchmarkMessage`를 제공합니다. `GOOD`의 `positiveFeedback`은 `null`입니다.

## 한국인 오류 패턴 seed

1차 구현에서는 한국인 학습자 오류 패턴 데이터를 AI 서버 seed로 관리합니다. seed 파일은 `app/data/error_patterns.json`이고, catalog 로더는 `app/services/error_pattern_catalog.py`입니다.

턴 피드백 LLM은 외부 응답 필드와 함께 내부 메타데이터인 `detectedPatterns`를 반환할 수 있습니다. AI 서버는 이 값을 `TurnFeedbackData` 검증 전에 분리해 캐시에만 저장하고, 백엔드 응답에는 노출하지 않습니다.

`breaks_meaning=false`인 관사, 시제, 복수, be 생략, 주어-동사 일치는 의미가 통하면 교정 폭격 대신 `benchmarkMessage`와 `highlightMessage`의 게임화 소재로 씁니다. `breaks_meaning=true`인 Konglish, 어휘 선택, 주어·목적어 생략은 `NEEDS_IMPROVEMENT`의 우선 교정 후보로 둡니다.

`detectedPatterns`는 내부 점수 계산에도 반영됩니다. 어려운 구조를 시도한 경우 문장 복잡도에 가산하고, 의미를 깨는 오류는 이해 가능성에서 더 크게 감점합니다.

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
