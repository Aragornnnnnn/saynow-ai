# SayNow AI 서버 — 2차 MVP TODO

## 1. API 전환

- [x] `POST /api/v1/conversation/next-question` 추가.
- [x] `POST /api/v1/conversation/feedback` 추가.
- [x] 1차 MVP `turn-evaluations`, `session-feedbacks`, `stt`, `tts`, `scenarios` 라우터 제거.

## 2. 책임 분리

- [x] AI 서버에서 오디오 업로드 제거.
- [x] AI 서버에서 Whisper STT 호출 제거.
- [x] AI 서버에서 OpenAI TTS 호출 제거.
- [x] 세션 완료 판정과 누적 슬롯 저장을 백엔드 책임으로 분리.

## 3. 꼬리 질문 생성

- [x] 백엔드가 넘긴 `slots` 기준으로 미충족 슬롯만 판단.
- [x] 이번 발화로 새롭게 충족된 슬롯만 `filledSlots`에 반환.
- [x] 이미 채워진 슬롯은 응답에서 제외.
- [x] 남은 미충족 슬롯이 없으면 `nextQuestion`, `translatedQuestion`을 `null`로 반환.

## 4. 대화 피드백 생성

- [x] 완료된 세션의 `turns[]` 텍스트 목록을 직접 수신.
- [x] 전체 이해도와 총평 반환.
- [x] 턴별 `turnId`, `feedbackRequired`, `nativeUnderstanding`, `nativeLanguageInterpretation`, `betterExpression` 반환.
- [x] `turnId`가 요청과 동일한 순서로 보존되는지 검증.

## 5. 검증

- [x] 새 서비스 계약 테스트 추가.
- [x] 새 라우터 계약 테스트 추가.
- [x] 기존 1차 MVP 테스트 제거.
- [x] 전체 테스트 실행.
