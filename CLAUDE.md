# SayNow AI Server — 2차 MVP

## Project Overview

SayNow 백엔드가 호출하는 내부 AI 서버입니다. 2차 MVP에서는 오디오, STT, TTS, 세션 상태 저장을 담당하지 않고, 백엔드가 전달한 텍스트와 슬롯 상태를 바탕으로 꼬리 질문과 최종 피드백만 생성합니다.

## Tech Stack

- **Framework**: FastAPI.
- **LLM**: OpenAI Chat Completions.
- **Validation**: Pydantic v2.
- **Runtime State**: 없음. 요청마다 필요한 컨텍스트를 백엔드가 전달합니다.

## Project Structure

```text
saynow-ai/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   └── routes/
│   │       └── conversation.py
│   ├── services/
│   │   └── conversation_service.py
│   ├── core/
│   │   ├── llm.py
│   │   └── logger.py
│   └── models/
│       └── conversation.py
├── tests/
│   ├── test_conversation_routes.py
│   └── test_conversation_service.py
├── requirements.txt
└── readme.md
```

## API

### `POST /api/v1/conversation/next-question`

- 백엔드가 직전 질문, 사용자 텍스트 발화, 시나리오 제목과 목표, 현재 슬롯 상태를 전달합니다.
- AI 서버는 이번 발화로 새롭게 충족된 슬롯만 `filledSlots`에 반환합니다.
- 이미 `filled=true`인 슬롯은 `filledSlots`에 다시 넣지 않습니다.
- 모든 미충족 슬롯이 이번 발화로 채워졌다면 `nextQuestion`과 `translatedQuestion`은 `null`입니다.
- 세션 완료 여부와 누적 슬롯 상태 갱신은 백엔드 책임입니다.

### `POST /api/v1/conversation/feedback`

- 백엔드가 완료된 세션의 턴 목록을 텍스트로 전달합니다.
- AI 서버는 전체 이해도, 총평, 턴별 피드백을 반환합니다.
- `turnId`는 백엔드 매핑을 위해 요청값과 동일하게 보존해야 합니다.
- 잘한 응답은 내부 점수 `85-100`이면서 질문 의도 답변, 턴 목표 충족, 추가 추측 없는 이해, 의미 차단 오류 없음 조건을 모두 만족할 때만 `feedbackRequired=false`로 반환합니다.
- `betterExpression`은 사용자 발화의 의도, 단어 수준, 문장 형태를 유지하면서 딱 한 단계만 개선해야 합니다.
- LLM 호출은 같은 입력에 같은 기준을 적용하기 위해 `temperature=0`을 사용합니다.

## Error Policy

- 잘못된 요청은 HTTP 400과 `{"code": "INVALID_REQUEST", "message": "잘못된 요청입니다."}`를 반환합니다.
- LLM 호출 실패나 계약에 맞지 않는 LLM 응답은 HTTP 500과 `AI_GENERATION_FAILED`를 반환합니다.

## Environment Variables

```bash
OPENAI_API_KEY=
LOG_LEVEL=INFO
```

## Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Verification

```bash
OPENAI_API_KEY=test-key python -m unittest discover -s tests -p 'test*.py'
```
