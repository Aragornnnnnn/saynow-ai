# SayNow AI 서버 — 3차 MVP TODO

## 1. API 전환

- [x] `POST /api/v1/conversation/next-question`를 4개 고정 질문 프리톡 계약으로 전환.
- [x] `POST /api/v1/conversation/turn-feedback` 추가.
- [x] `POST /api/v1/conversation/session-feedback` 추가.
- [x] 2차 MVP `feedback`, `feedback/stream`, 슬롯 판정 계약 제거.

## 2. 책임 분리

- [x] AI 서버에서 다음 질문 선택과 슬롯 완료 판정 제거.
- [x] 백엔드가 전달한 다음 고정 질문을 기준으로 맞장구와 질문 연결만 생성.
- [x] 턴별 피드백은 AI 서버 캐시에 저장.
- [x] 최종 피드백은 캐시된 턴별 피드백을 모아 생성.

## 3. 피드백 품질

- [x] `GOOD`과 `NEEDS_IMPROVEMENT` 타입 분리.
- [x] 모든 턴별 피드백에 한국어 비유 포함.
- [x] 잘한 발화는 억지로 고치지 않고 칭찬 요약과 이유를 반환.
- [x] 고쳐야 하는 발화는 핵심 포인트, 이유, +1 개선 표현을 반환.
- [x] 세션 최종 피드백에 native score, 수준 라벨, 총평 포함.

## 4. 제거

- [x] Assistance RAG 코드와 임베딩 클라이언트 제거.
- [x] 슬롯 모델, evidence policy, `filledSlots`, `turnClassification` 제거.
- [x] README와 작업 안내 문서 갱신.

## 5. 검증

- [x] 새 서비스 계약 테스트 추가.
- [x] 새 라우터 계약 테스트 추가.
- [x] 전체 테스트 실행.
- [x] `compileall` 실행.
- [x] `git diff --check` 실행.
