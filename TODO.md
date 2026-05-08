# SayNow AI 서버 — 백엔드 API 명세 대응 TODO

## 1. 엔드포인트 재설계

- [x] `POST /conversation/start` 제거
- [x] `POST /conversation/next` 제거
- [x] `GET /feedback/{session_id}` 제거
- [x] `POST /api/v1/turn-evaluations` 신규 생성
- [x] `POST /api/v1/session-feedbacks` 신규 생성

## 2. 세션 관리 제거

- [x] `conversation_service.py`의 `_sessions` in-memory 딕셔너리 제거
- [x] `start_session`, `next_turn`, `get_session` 함수 제거
- [x] AI 서버를 stateless하게 재설계 (요청마다 컨텍스트를 받아 처리)

## 3. STT 통합

- [x] `/turn-evaluations`에서 audio 파일(multipart) 직접 수신
- [x] 요청 내에서 Whisper STT 호출 후 `transcript` + `sttConfidence` 생성
- [ ] 기존 `POST /stt` 별도 엔드포인트 제거 (또는 유지 여부 백엔드와 협의)

## 4. 턴 평가 요청 모델 변경

- [x] 요청 필드 추가: `scenarioSituation`, `scenarioGoal`, `currentQuestion`
- [x] 요청 필드 추가: `filledSlots: [{slotKey, slotValue}]` (Spring이 누적 관리해서 전달)
- [x] 요청 필드 추가: `conversationHistory: [{role, content}]`

## 5. 턴 평가 응답 모델 변경

- [x] 응답 필드 추가: `transcript` (STT 결과)
- [x] 응답 필드 추가: `sttConfidence` (float, 0~1)
- [x] 응답 필드 추가: `scenarioStatus` (IN_PROGRESS | SUCCESS | FAILURE)
- [x] 응답 필드 변경: `filledSlots: [{slotKey, slotValue}]` (이번 턴에 새로 채워진 슬롯)
- [x] 응답 필드 변경: `nextQuestion: {questionText, ttsAudio}` (객체로 변경)
- [x] 응답 필드 변경: `resultMessage: {messageText, ttsAudio}` (SUCCESS/FAILURE 시 반환)

## 6. 슬롯 기반 클리어 판단 로직 변경

- [x] 기존 `_check_cleared` (required_info 문자열 리스트 평가) 제거
- [x] 신규: 이번 턴 발화에서 새로 채워진 slotKey/slotValue 추출하는 LLM 프롬프트 작성
- [x] 신규: 모든 슬롯이 채워졌는지 판단해 `scenarioStatus` 결정하는 로직 작성

## 7. 피드백 요청 모델 변경

- [x] 요청 구조 변경: `session_id` 기반 조회 → `turns[]` 배열 직접 수신
- [x] 요청 필드: `scenarioId`, `scenarioGoal`, `turns: [{transcript, question, responseTimeSec}]`

## 8. 피드백 응답 모델 변경

- [x] 응답 필드 추가: `summary` (한글 전체 요약)
- [x] 응답 필드 추가 (turn별): `scoreDelta` (이전 턴 대비 점수 변화)
- [x] 응답 필드 추가 (turn별): `improvedUnderstoodScore` (betterExpression 사용 시 예상 점수)
- [x] 응답 필드 추가 (turn별): `reason` (한글 피드백 이유)
- [x] 기존 `fail_reason` 제거 (turn별 `reason`으로 대체)

## 9. TTS 필드명 변경

- [x] 응답의 `audio_base64` → `ttsAudio` 로 필드명 통일

## 10. Pydantic 모델 정리

- [x] `StartRequest`, `StartResponse`, `StartResponseData` 제거
- [x] `NextRequest`, `NextResponse`, `NextResponseData` 제거
- [x] `TurnEvaluationRequest`, `TurnEvaluationResponse` 신규 작성
- [x] `SessionFeedbackRequest`, `SessionFeedbackResponse` 신규 작성
- [x] `FeedbackData`, `Utterance` 모델 신규 필드에 맞게 수정
