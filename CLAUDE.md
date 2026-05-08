# SayNow — English Speaking Practice App (MVP)

## Project Overview
영어 스피킹 연습 앱. 시나리오 기반 롤플레이 → AI 실시간 대화 → 피드백의 3단계 흐름.

## Tech Stack
- **Backend**: FastAPI (Python)
- **LLM**: Anthropic Claude API (claude-sonnet-4-6)
- **STT**: OpenAI Whisper API
- **TTS**: OpenAI TTS API
- **Data**: JSON 기반 시나리오 파일 (MVP, DB 없음)

## Project Structure
```
saynow/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # env vars & settings
│   ├── api/
│   │   └── routes/
│   │       ├── scenario.py      # GET /scenarios, GET /scenarios/{id}
│   │       ├── conversation.py  # POST /conversation/start, /next, /evaluate
│   │       ├── stt.py           # POST /stt  (audio → text)
│   │       ├── tts.py           # POST /tts  (text → audio)
│   │       └── feedback.py      # GET /feedback/{session_id}
│   ├── services/
│   │   ├── scenario_service.py   # 시나리오 데이터 로드 & 조회
│   │   ├── conversation_service.py # 대화 흐름 관리, 클리어 판단
│   │   ├── stt_service.py        # Whisper 호출
│   │   ├── tts_service.py        # OpenAI TTS 호출
│   │   └── feedback_service.py   # 이해도 분석, 피드백 생성
│   ├── core/
│   │   └── llm.py               # Anthropic client 초기화 & 공통 호출
│   ├── models/
│   │   ├── scenario.py          # Pydantic models
│   │   ├── conversation.py
│   │   └── feedback.py
│   └── data/
│       └── scenarios.json       # 5개 카테고리 × 2개 시나리오
├── requirements.txt
├── .env.example
└── readme.md
```

## Core Business Logic

### 시나리오 구조
- 카테고리: airport / hotel / cafe / restaurant / taxi (각 2개)
- 각 시나리오: id, category, title, situation, goal, required_info[], max_questions(5)

### 대화 흐름 (conversation_service)
1. `start`: 세션 생성, 첫 AI 발화 생성 (TTS 포함)
2. `next`: 유저 STT 결과 받아 → 이해도 분석 → 꼬리질문 생성 → 클리어 여부 판단
3. 클리어 조건: required_info 모두 충족 OR 질문 5회 소진(실패)
4. 세션 상태는 인메모리 dict로 관리 (MVP)

### 이해도 분석 (LLM 프롬프트)
- 입력: 유저 발화(STT 텍스트), 시나리오 컨텍스트
- 출력: comprehension_score(0~100), native_perception(미국인 귀에 들린 내용), better_expression

### 피드백 구조
- total_comprehension: 전체 평균
- utterances[]: { text, response_time_sec, comprehension_score, native_perception, better_expression }

## API Conventions
- 모든 응답: `{ success: bool, data: ..., error: str | null }`
- 오디오 업로드: multipart/form-data
- 오디오 반환: base64 encoded string (MVP)

## Environment Variables
```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

## Development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
