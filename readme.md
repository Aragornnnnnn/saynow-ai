# SayNow AI Server

3차 MVP 백엔드가 호출하는 내부 AI 서버입니다.

## 역할

- 직전 사용자 발화에 대한 짧은 맞장구와 백엔드가 전달한 다음 고정 질문을 하나의 `aiQuestion`으로 연결합니다.
- `next-question` 응답에는 상대 역할 기준의 속마음인 `innerThought`, `innerThoughtType`을 함께 내려줍니다.
- 사용자 발화 1개에 대한 턴별 피드백을 생성하고 AI 서버 프로세스 메모리 캐시에 최대 3시간 보관합니다.
- 최종 피드백 생성 시 캐시된 턴별 피드백을 모아 `nativeScore`, `highlightMessage`와 함께 반환합니다.
- 영어 학습 가이드 질문은 기존 `guide` API로 계속 처리합니다.

슬롯 완료 판정, 세션/턴 생성, DB 저장, NPS, 최종 완료 상태 관리는 백엔드 책임입니다.

## API

### `POST /api/v1/conversation/next-question`

백엔드가 다음 고정 질문을 `nextQuestion`으로 전달하면 AI 서버는 직전 발화에 대한 짧은 반응과 해당 고정 질문을 자연스럽게 이어 붙입니다. AI 서버는 다음 질문을 새로 고르지 않습니다.
요청의 `scenario.counterpartRole`은 필수입니다. 같은 발화라도 교수, 친구, 룸메이트, 직원 역할에 따라 속마음이 달라질 수 있기 때문입니다.

응답.

```json
{
  "aiQuestion": "Oh, you like spicy pizza. Do you cook often?",
  "translatedQuestion": "매운 피자를 좋아하는군요. 요리는 자주 하나요?",
  "innerThought": "이렇게 이유까지 말해주니까 대화하기 편하네.",
  "innerThoughtType": "GOOD"
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
  "highlightMessage": "한국인의 79%가 틀리는 a/an을 정확히 쓴 사람",
  "turnFeedbacks": [
    {
      "turnId": 5000,
      "feedbackType": "GOOD",
      "koreanAnalogy": "\"저는 피자가 좋아요. 매워서요\"라고 자연스럽게 이유를 붙여 말하는 것과 같아요.",
      "feedbackDetail": "이유를 because로 자연스럽게 붙였고, 좋아하는 음식과 이유를 한 문장 안에서 분명하게 연결했어요.",
      "positiveFeedback": null,
      "correctionExpression": null,
      "correctionReason": null,
      "benchmarkMessage": "한국인의 79%가 틀리는 a/an을 정확히 썼어요"
    },
    {
      "turnId": 5001,
      "feedbackType": "NEEDS_IMPROVEMENT",
      "koreanAnalogy": "\"그게 뭔지 모르겠어\"라고 말하려다 어순이 살짝 꼬인 문장으로 말하는 것과 같아요.",
      "positiveFeedback": "어려운 간접의문문 구조에 도전한 점이 좋아요. 틀렸더라도 그 시도 자체가 다음 단계로 가는 재료예요.",
      "feedbackDetail": null,
      "correctionExpression": "I do not know what it is.",
      "correctionReason": "what is it → what it is. 간접의문문에서는 의문문 어순이 아니라 평서문 어순을 써야 해요.",
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

`highlightMessage`는 전체 총평이 아니라 발화별 피드백을 열어 보게 만드는 칭호형 후킹 문구입니다. 우선순위는 사용자가 잘한 GOOD 정량 포인트입니다. 예를 들어 `한국인의 79%가 틀리는 a/an을 정확히 쓴 사람`처럼 마침표 없는 명사구를 우선합니다. 이런 GOOD 포인트가 없으면 `한국인 40%가 헷갈리는 간접의문문에 도전한 사람`처럼 NEEDS_IMPROVEMENT의 도전 포인트를 fallback hook으로 씁니다.

`koreanAnalogy`는 문법 설명이 아니라 원래 영어가 한국어 감각으로 어떻게 들리는지 보여주는 필드입니다. `한국어로 비유하자면`, `한국어로 치면` 같은 접두어 없이 `"..."라고 ...하는 것과 같아요.`처럼 바로 본론으로 시작합니다. raw JSON에서는 문자열 안 큰따옴표가 `\"`로 escape되지만, 클라이언트에서 JSON을 파싱해 렌더링하면 역슬래시는 보이지 않습니다.

`innerThought`는 피드백 설명문이 아니라 상대 역할의 1인칭 속마음입니다. 예를 들어 친구에게는 차갑게 들리는 말도 교수에게는 무례하거나 명령처럼 들릴 수 있습니다. `innerThoughtType`은 `GOOD`, `NORMAL`, `BAD` 중 하나입니다.

`NEEDS_IMPROVEMENT`에는 `koreanAnalogy`, `positiveFeedback`, `correctionExpression`, `correctionReason`을 반드시 포함합니다. `correctionExpression`은 개선된 영어 표현만 담고, `correctionReason`은 `what is it → what it is`처럼 가장 짧은 의미 단위의 before→after와 한국어 이유를 담습니다. `feedbackDetail`과 `benchmarkMessage`는 `null`로 둡니다. `GOOD`에는 `koreanAnalogy`, `feedbackDetail`, `benchmarkMessage`를 반드시 포함하고, `positiveFeedback`, `correctionExpression`, `correctionReason`은 `null`입니다. 검증된 정량 패턴이 있으면 catalog 의미를 쓰고, 없으면 사용자 발화의 surface usage를 보고 기존 수치 catalog hook을 느슨하게 재활용합니다. 턴별 `benchmarkMessage`는 `한국인의 37%가 놓치는 복수형 명사+s를 빠짐없이 챙겼어요`처럼 문장형으로 내려가고, 세션 `highlightMessage`는 `한국인의 37%가 놓치는 복수형 명사+s를 빠짐없이 챙긴 사람`처럼 칭호형으로 유지합니다. 이 값은 엄밀한 오류 진단이 아니라 재미용 학습 hook입니다.

## 한국인 오류 패턴 seed

1차 구현에서는 한국인 학습자 오류 패턴 데이터를 AI 서버 seed로 관리합니다. seed 파일은 `app/data/error_patterns.json`이고, catalog 로더는 `app/services/error_pattern_catalog.py`입니다.

턴 피드백 LLM은 외부 응답 필드와 함께 내부 메타데이터인 `detectedPatterns`를 반환할 수 있습니다. AI 서버는 이 값을 `TurnFeedbackData` 검증 전에 분리해 캐시에만 저장하고, 백엔드 응답에는 노출하지 않습니다.

`breaks_meaning=false`인 관사, 시제, 복수, be 생략, 주어-동사 일치는 의미가 통하면 교정 폭격 대신 `benchmarkMessage`와 `highlightMessage`의 게임화 소재로 씁니다. `GOOD`에서는 `indirect_question_word_order`, `article_a_omission`, `article_the`, `noun_plural`, `sv_agreement`, `be_omission`, `prep_omission`, `tense_aspect` 순서로 surface usage를 감지해 수치형 hook을 만듭니다. `breaks_meaning=true`인 Konglish, 어휘 선택, 주어·목적어 생략은 `NEEDS_IMPROVEMENT`의 우선 교정 후보로 둡니다.

`detectedPatterns`는 내부 점수 계산에도 반영됩니다. 어려운 구조를 시도한 경우 문장 복잡도에 가산하고, 의미를 깨는 오류는 이해 가능성에서 더 크게 감점합니다.

## Error Policy

- 잘못된 요청은 HTTP 400과 `{"code": "INVALID_REQUEST", "message": "잘못된 요청입니다."}`를 반환합니다.
- 필요한 턴별 피드백이 아직 없으면 HTTP 409와 `TURN_FEEDBACK_NOT_READY`를 반환합니다.
- LLM 호출 실패나 계약에 맞지 않는 LLM 응답은 HTTP 500과 `AI_GENERATION_FAILED`를 반환합니다.

## Environment Variables

```bash
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_NEXT_QUESTION_MODEL=gpt-5.4-mini
OPENAI_TURN_FEEDBACK_MODEL=gpt-5.4-mini
OPENAI_SESSION_FEEDBACK_MODEL=gpt-5.4-mini
OPENAI_FALLBACK_MODEL=gpt-4o-mini
LLM_REQUEST_TIMEOUT_SECONDS=60
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
