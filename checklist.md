# 작업 체크리스트

## 한국어 비유 품질 개선

- [x] 어색한 sushi/free-time 한국어 비유를 재현하는 RED 테스트를 추가한다.
- [x] `koreanAnalogy` 프롬프트 기준을 “어색한 한국어 예시 + 짧은 느낌 설명”으로 보강한다.
- [x] sushi와 free-time 후처리를 새 기준에 맞게 개선한다.
- [x] focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] GitHub 이슈 체크리스트를 갱신한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## GitHub Wiki 3차 MVP 최신화

- [x] GitHub Wiki repo를 별도 clone한다.
- [x] 기존 Wiki 문서와 현재 3차 MVP 코드 계약 차이를 확인한다.
- [x] API Reference, Home, Release Notes, Sidebar를 최신 계약에 맞게 갱신한다.
- [x] humanizer 기준으로 한국어 문장을 점검한다.
- [x] Wiki 변경분을 커밋하고 push한다.
- [x] 본 repo 작업 기록을 커밋한다.

## 최신 커밋 배포 후 현재 시나리오 재검증

- [x] 최신 커밋을 `origin/develop`에 push한다.
- [x] develop AI 배포 GitHub Actions run 성공을 확인한다.
- [x] 배포 후 `/health`와 `/openapi.json`에서 최신 필드 계약을 확인한다.
- [x] 현재 시나리오 3개, 12턴 live smoke를 다시 실행한다.
- [x] 배포 후 결과를 Obsidian 문서에 추가한다.
- [x] 테스트와 diff check를 확인한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## 최신 현재 시나리오 데이터 재검증과 문서화

- [x] 첨부 시나리오 데이터와 기존 결과 JSON을 다시 확인한다.
- [x] 최신 기준으로 현재 시나리오 3개, 12턴 smoke를 실행한다.
- [x] 이전 전체 smoke와 최신 smoke의 차이를 비교한다.
- [x] `humanizer` 기준으로 Obsidian 문서에 결과를 정리한다.
- [x] 검증 명령과 작업 결과를 확인한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## GOOD/NEEDS 분류 기준 정립

- [x] `GOOD`과 `NEEDS_IMPROVEMENT`의 판단 기준을 `context-notes.md`에 기록한다.
- [x] 짧지만 명확한 답변이 세부 정보 부족만으로 NEEDS가 되지 않는 기준을 테스트로 고정한다.
- [x] 명백한 문법 문제를 모델이 GOOD으로 내려도 NEEDS로 보정하는 RED 테스트를 추가한다.
- [x] 무례하거나 방어적으로 들리는 표현을 모델이 GOOD으로 내려도 NEEDS로 보정하는 RED 테스트를 추가한다.
- [x] `prompt-engineering-patterns` 기준으로 턴 피드백 프롬프트의 역할, 판단 기준, self-check, 구조화 출력을 점검한다.
- [x] 턴 피드백 프롬프트와 고신뢰 후처리를 기준에 맞게 수정한다.
- [x] focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## 질문 수 변동 대응 세션 점수/라벨 기준

- [x] `GOOD` 비율 기반 점수/라벨 기준을 `context-notes.md`에 기록한다.
- [x] 3문항, 4문항, 5문항 세션 점수/라벨 RED 테스트를 추가한다.
- [x] LLM 점수 clamp와 서버 라벨 덮어쓰기 RED 테스트를 추가한다.
- [x] 세션 피드백 프롬프트에서 LLM과 서버의 역할 분리를 명확히 한다.
- [x] 서버 후처리를 질문 수 고정이 아닌 `GOOD` 비율 기반으로 바꾼다.
- [x] focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] 현재 시나리오 smoke를 실행한다.
- [x] Obsidian 문서에 결과를 정리한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## 현재 시나리오 품질 개선 후보 보정

- [x] sleeping habit GOOD 피드백이 발화 내용에 맞게 grounding되는 RED 테스트를 추가한다.
- [x] tteokbokki GOOD 피드백이 사용자가 말하지 않은 감정을 보태지 않는 RED 테스트를 추가한다.
- [x] session-feedback 총평의 문서체와 번역투를 줄이는 RED 테스트를 추가한다.
- [x] travel next-question의 반복적인 `fun trip` 맞장구를 사용자 발화 기반 표현으로 바꾸는 RED 테스트를 추가한다.
- [x] 최소 후처리와 프롬프트 보강으로 테스트를 GREEN으로 만든다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 실제 모델 품질 스모크를 재실행해 개선 결과를 확인한다.

## 현재 시나리오 데이터 품질 테스트와 문서화

- [x] 첨부된 현재 시나리오 데이터의 파싱 조건을 확인한다.
- [x] 기존 `turn-feedback` 필드 단순화 변경을 의미 있는 단위로 커밋했는지 확인한다.
- [x] 실제 develop 설정으로 3개 시나리오 12개 턴 품질 스모크를 실행한다.
- [x] 각 턴의 `turn-feedback`, 고정 질문 연결 `next-question`, 캐시 기반 `session-feedback` 결과를 수집한다.
- [x] 품질 이슈와 개선 후보를 사용자 검토용 표로 정리한다.
- [x] `humanizer` 기준으로 설명 문장을 점검해 Obsidian 3차 MVP 하위 문서에 기록한다.
- [x] 새 계약과 충돌하는 기존 live smoke 문서를 삭제하고 링크를 정리한다.

## turn-feedback 필드 단순화

- [x] `feedbackDetail` 중심 계약을 검증하는 RED 테스트를 추가한다.
- [x] `TurnFeedbackData` 모델을 `feedbackType`, `koreanAnalogy`, `feedbackDetail`, `betterExpression` 구조로 단순화한다.
- [x] `turn-feedback` 프롬프트와 self-check를 새 필드 계약에 맞춘다.
- [x] 기존 후처리와 세션 총평 보정이 새 필드를 사용하도록 바꾼다.
- [x] route/service focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] 변경 이유와 검증 결과를 `context-notes.md`에 기록한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## 3차 MVP AI 계약 구현

- [x] 3차 MVP 기준 문서에서 AI 서버가 맡는 범위를 확정한다.
- [x] 기존 2차 MVP 슬롯 기반 `next-question` 계약과 통합 `feedback` 계약을 제거 대상으로 분류한다.
- [x] `/api/v1/conversation/next-question` 문서 계약 테스트를 RED로 추가한다.
- [x] `/api/v1/conversation/turn-feedback` 문서 계약 테스트를 RED로 추가한다.
- [x] `/api/v1/conversation/session-feedback` 문서 계약 테스트를 RED로 추가한다.
- [x] 다음 질문 생성 프롬프트를 “맞장구 + BE 고정 질문 연결” 기준으로 단순화한다.
- [x] 턴별 피드백 생성 프롬프트를 `GOOD` / `NEEDS_IMPROVEMENT` 품질 기준으로 구현한다.
- [x] 세션 최종 피드백 프롬프트를 캐시된 턴별 피드백 종합 기준으로 구현한다.
- [x] AI 쪽 턴 피드백 캐시를 추가하고 최종 피드백에서 `expectedTurnIds` 누락을 409로 처리한다.
- [x] 더 이상 쓰지 않는 슬롯 모델, 슬롯 판정 로직, RAG 보조 로직, 통합 피드백 SSE 경로를 제거한다.
- [x] README와 파일 헤더를 3차 MVP 기준으로 갱신한다.
- [x] focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] 변경 이유와 검증 결과를 `context-notes.md`에 기록한다.
- [x] 의미 있는 단위로 커밋한다.

## 3차 MVP 실제 모델 품질 스모크 보정

- [x] 실제 `gpt-4o-mini` develop 설정으로 3차 MVP 품질 스모크를 실행한다.
- [x] 다음 질문이 고정 질문만 반환될 때 짧은 맞장구를 보강하는 RED 테스트를 추가한다.
- [x] 명확한 `because` 답변을 세부 정보 부족만으로 과교정하지 않는 RED 테스트를 추가한다.
- [x] `plusOneExpression`이 사용자 의도와 다른 새 문장을 만들 때 같은 발화의 교정문으로 보정하는 RED 테스트를 추가한다.
- [x] 세션 총평이 영어로 내려올 때 한국어 fallback을 적용하는 RED 테스트를 추가한다.
- [x] 프롬프트와 후처리 품질 가드를 최소 범위로 보강한다.
- [x] focused 테스트, 대화 서비스 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] 실제 모델 품질 스모크를 재실행한다.
- [x] `GOOD`과 `NEEDS_IMPROVEMENT` 타입별 필드가 섞이지 않도록 검증을 강화한다.
- [x] 일반 맞장구(`I see.`, `That's great to hear!`)를 사용자 발화 기반 맞장구로 보정한다.
- [x] GOOD 피드백의 칭찬 설명이 영어로 오면 한국어 설명으로 보정한다.
- [x] 실제 모델 10개 대표 케이스 평가를 실행한다.
- [x] `prompt-engineering-patterns` 기준으로 next-question 프롬프트를 재점검한다.
- [x] 맞장구 기준을 발화 인용이 아니라 실제 대화감으로 재정의한다.
- [x] 테스트가 실제 피드백 출력 내용을 함께 검증하고 보고하도록 갱신한다.
- [x] 실제 모델 10개 대표 케이스 평가를 새 기준으로 재실행한다.

## develop/main EC2 배포 대상 매핑 복구

- [x] workflow 매핑 회귀 테스트를 현재 운영 IP 기준으로 수정한다.
- [x] `develop` 브랜치는 develop GitHub environment와 `/saynow/develop` SSM 경로를 사용한다.
- [x] `main` 브랜치는 prod GitHub environment와 `/saynow/prod` SSM 경로를 사용한다.
- [x] SSH key raw/base64 처리 차이로 main 배포가 깨지지 않게 prod workflow 키 처리를 보강한다.
- [x] workflow 테스트, YAML 파싱, diff check를 실행한다.
- [x] 변경 이유와 검증 결과를 `context-notes.md`에 기록한다.
- [x] 변경 사항을 커밋한다.

## turn-feedback turnId 불일치 보정

- [x] 요청 `turnId=3`인데 모델이 `turnId=5000`을 반환하는 RED 테스트를 추가한다.
- [x] LLM 응답의 `turnId`를 서버 요청값으로 덮어써 캐시와 응답 식별자를 고정한다.
- [x] turn-feedback 프롬프트 schema에서 고정 `turnId=5000` 예시를 제거한다.
- [x] focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] 변경 이유와 검증 결과를 `context-notes.md`에 기록한다.

## 3차 MVP BE live smoke 품질 보정

- [x] BE live smoke 결과에서 세션 점수, GOOD 칭찬, NEEDS 비유, 교정 표현의 품질 실패를 회귀 테스트로 고정한다.
- [x] `prompt-engineering-patterns` 기준으로 턴 피드백과 세션 피드백 프롬프트의 역할 분리, grounding, self-check를 보강한다.
- [x] 모델 응답이 품질 기준을 벗어나도 서버 후처리에서 좁게 보정한다.
- [x] focused 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] 실제 AI 서버 또는 실제 모델 경로로 대표 케이스를 직접 평가한다.
- [x] 개선 과정과 테스트 결과를 `context-notes.md`와 사용자 보고용 표로 정리한다.

## REPEAT_REQUEST 오분류 방어

- [x] `ABC`, `haha`가 모델 raw `REPEAT_REQUEST`여도 최종 `INVALID_RESPONSE`가 되는 RED 테스트를 추가한다.
- [x] 정상 반복 요청 fast-path가 모델 호출 없이 `REPEAT_REQUEST`를 유지하는지 확인한다.
- [x] raw `REPEAT_REQUEST`를 deterministic repeat detector로 재검증하도록 구현한다.
- [x] next-question prompt에 repeat self-check를 최소 보강한다.
- [x] focused 테스트와 전체 검증을 실행한다.
- [x] Obsidian 품질 회귀 문서에 세션 351 원인과 수정 기준을 기록한다.
- [x] 변경 사항을 커밋하고 develop에 push한다.
- [x] GitHub Actions develop 배포와 `/health`를 확인한다.
- [x] 배포 후 direct AI smoke로 `ABC`, `haha`, 정상 반복 요청을 확인한다.

## AI 슬롯명 노출과 피드백 문맥 오염 수정

- [x] `missed_connection` fallback 질문/번역이 슬롯명이나 readable slot phrase를 노출하는 RED 테스트를 추가한다.
- [x] 알 수 없는 semantic slot fallback이 raw slot key 대신 generic safe question을 쓰는 RED 테스트를 추가한다.
- [x] `next-question` visible field self-check와 postprocess repair를 함께 적용한다.
- [x] DB 수정 후 기준 `baggage_issue_detail` 회귀 테스트를 추가한다.
- [x] baggage 질문의 `I don't know`가 주문 문맥으로 오염되는 RED 테스트를 추가한다.
- [x] order 질문의 `I don't know`는 기존 주문 문맥을 유지하는 회귀 테스트를 추가한다.
- [x] 동일 `originalQuestion + userUtterance` 입력의 `feedbackRequired` 혼합 판정을 repair 대상으로 감지한다.
- [x] focused RED/GREEN 테스트를 실행한다.
- [x] 전체 unittest, compileall, diff check를 실행한다.
- [x] Obsidian 품질 회귀 문서에 변경 이유와 검증 결과를 정리한다.
- [x] 변경 사항을 의미 있는 단위로 커밋하고 develop에 push한다.
- [x] GitHub Actions develop 배포와 `/health`를 확인한다.
- [x] 배포 후 direct AI smoke를 실행한다.

## AI 대화 턴 지연 3차 개선

- [x] deterministic completion skip reason 로그를 추가한다.
- [x] target request slot local accept로 semantic verifier 호출을 보수적으로 줄인다.
- [x] `next_options_request` 최종 완료 fast-path를 실제 슬롯명으로 고정한다.
- [x] 사용자 경험 악영향 가능성이 있는 semantic evidence timeout fail-closed 변경을 제거한다.
- [x] 일반 슬롯 답변 경로의 next-question prompt와 max token을 줄인다.
- [x] 관련 테스트, 전체 unittest, compileall, diff check를 실행한다.
- [x] Obsidian 병목 제거 문서에 변경 이유와 검증 결과를 이어서 정리한다.

## fallback 질문/번역 슬롯명 노출 제거

- [x] baggage 계열 fallback 질문/번역이 슬롯명을 그대로 노출하는 RED 테스트를 추가한다.
- [x] 슬롯명 대신 slot description/hints 기반 자연스러운 질문을 반환하도록 수정한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] Obsidian 병목 제거 문서에 원인과 배포 후 확인 결과를 정리한다.
- [x] 변경 사항을 의미 있는 단위로 커밋하고 develop에 push한다.

## Assistance RAG 기본 비활성화

- [x] `assistance_rag_enabled` 기본값이 false인 RED 테스트를 추가한다.
- [x] 명시적으로 `assistance_rag_enabled=true`와 DB URL을 줄 때만 pgvector store를 만드는 테스트를 추가한다.
- [x] 설정 기본값을 false로 바꾸고 기존 RAG 구현은 rollback용으로 유지한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] `SayNow AI 병목 제거 작업 2026-05-31` 하위 문서에 RAG 비활성화 이유와 확인 방법을 정리한다.
- [x] 변경 사항을 커밋하고 develop에 push한다.
- [x] 재배포 후 direct AI 또는 BE 재현으로 지연 변화를 확인한다.

## AI deterministic completion fast-path

- [x] 남은 요청형 슬롯 하나를 로컬 정책으로 채울 수 있는데도 main LLM을 호출하는 RED 테스트를 추가한다.
- [x] 해당 케이스에서 RAG lookup, main LLM, semantic verifier, RAG save를 모두 건너뛰도록 구현한다.
- [x] 슬롯명이나 airport 도메인에 의존하지 않고 `evidencePolicy`, target slot, request act 기준으로 처리한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] `SayNow AI 병목 제거 작업 2026-05-31` 하위 문서에 개선 이유와 기대 효과를 정리한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## repeat request 별도 분류와 빠른 재질문

- [x] `Parden Can you tell again?` 같은 반복 요청이 `REPEAT_REQUEST`가 되는 RED 테스트를 추가한다.
- [x] 반복 요청에서는 RAG lookup, main LLM, semantic verifier, RAG save가 호출되지 않게 구현한다.
- [x] 기존 `ASSISTANCE_REQUEST`, `INVALID_RESPONSE`, 슬롯 답변 회귀 테스트가 깨지지 않는지 확인한다.
- [x] `SayNow AI 병목 제거 작업 2026-05-31` 하위 문서에 분류 결정과 지연 개선 이유를 정리한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## repeat request 번역 원문 고정

- [x] `originalTranslatedQuestion`이 있을 때 `REPEAT_REQUEST`가 직전 한국어 번역을 그대로 반환하는 RED 테스트를 추가한다.
- [x] `NextQuestionRequest`에 optional `originalTranslatedQuestion`을 추가한다.
- [x] 반복 요청 fast-path와 모델 분류 후처리에서 직전 번역을 우선 사용하도록 구현한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## STT 의문문 request act 판정 확장

- [x] `?`가 없는 `Are there...`, `Is there...` 발화가 요청형 슬롯을 채우는 RED 테스트를 추가한다.
- [x] 슬롯명이나 airport 도메인에 의존하지 않는 공통 request-act 판정으로 구현한다.
- [x] 세션 271 반복 질문 원인과 개선 이유를 문서에 정리한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## 간접 옵션 요청 target-aware 분류

- [x] 직전 target이 요청형 슬롯이면 `I don't know what option I can do`가 `ANSWER`로 슬롯을 채우는 RED 테스트를 추가한다.
- [x] 직전 target이 다른 슬롯이면 같은 발화가 `ASSISTANCE_REQUEST`로 처리되고 슬롯을 채우지 않는 RED 테스트를 추가한다.
- [x] 슬롯명이나 airport 도메인에 의존하지 않는 target-aware request-slot 정책으로 구현한다.
- [x] BE 재현 결과와 결정 이유를 작업 문서에 정리한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## AI RAG lookup 조건부 실행

- [x] `ANSWER` 슬롯 답변에서 RAG lookup이 호출되지 않는 회귀 테스트를 RED로 확인한다.
- [x] 메뉴, 추천, 부가 정보 질문은 기존처럼 RAG lookup을 유지하는 회귀 테스트를 확인한다.
- [x] 슬롯 정책 기반으로 최신 발화가 미충족 슬롯 evidence에 가까우면 RAG lookup을 건너뛰도록 구현한다.
- [x] 관련 focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] `SayNow 응답 시간 및 지연 측정` 하위 문서에 변경 이유, 구현 결과, direct 검증 결과를 정리한다.
- [x] 변경 사항을 의미 있는 단위로 커밋하고 develop에 push한다.

## AI semantic evidence 지연 개선

- [x] `trace-test-002`, `trace-test-01`, `trace-test-010` 로그에서 postprocess 병목 패턴을 정리한다.
- [x] prompt-engineering-patterns 기준으로 marker 기반 fast-path의 확장성 한계를 재검토한다.
- [x] semantic evidence verifier를 슬롯별 N회 호출하지 않고 요청당 1회 batch 검증하도록 테스트로 고정한다.
- [x] 도메인 marker fast-path를 제거하고 구조화 JSON batch verifier로 `next_question_semantic_evidence` LLM 호출 수를 줄인다.
- [x] 기존 semantic evidence 회귀 테스트와 focused 테스트를 실행한다.
- [x] `SayNow 응답 시간 및 지연 측정` 하위 문서에 변경 과정과 검증 결과를 정리한다.

## AI 응답 시간 및 지연 측정

- [x] 요청 ID 전달과 workflow latency 로그 범위를 현재 코드에서 확인한다.
- [x] `X-Request-Id`가 AI 요청 컨텍스트에 저장되고 응답 헤더로 반환되는 RED 테스트를 추가한다.
- [x] 기존 단계별 timing 로그에 `requestId`가 포함되는 RED 테스트를 추가한다.
- [x] workflow 전체 소요 시간 로그가 남는 RED 테스트를 추가한다.
- [x] 요청 컨텍스트와 middleware를 최소 구현한다.
- [x] conversation service timing 로그에 `requestId`와 전체 workflow 로그를 추가한다.
- [x] focused 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## prompt-engineering-patterns 기준 보강

- [x] `next-question` 요청과 응답에 target slot metadata를 추가한다.
- [x] target slot이 있어도 최신 발화 근거가 있으면 여러 슬롯을 채울 수 있는 회귀 테스트를 추가한다.
- [x] live smoke에서 드러난 `exact dates` 기간 상세 재질문을 회귀 테스트로 고정하고 수정한다.
- [x] next-question few-shot 예시가 Output Schema와 같은 필드를 쓰는지 RED 테스트를 추가한다.
- [x] 한국어 slot description만으로도 duration intent를 판정하는 RED 테스트를 추가한다.
- [x] feedback self-check가 목적, 국가, 장소, 의도 hallucination을 다시 검증하는 RED 테스트를 추가한다.
- [x] 이름만 말한 발화의 betterExpression이 예시 답변임을 명확히 드러내는 RED 테스트를 추가한다.
- [x] 프롬프트와 후처리 보강을 구현한다.
- [x] 관련 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## scenario 207 슬롯 판정과 피드백 grounding 개선

- [x] 세션 207 문제를 작업 기록에 남긴다.
- [x] `Two week` 같은 짧은 기간 답변을 `stay_duration`으로 채우는 RED 테스트를 추가한다.
- [x] description/hints/question 기반 기간 evidence 보강을 구현한다.
- [x] 기간 답변 회귀 테스트와 관련 테스트를 통과시키고 커밋한다.
- [x] BE dev 케이스 4에서 target은 `next_options_request`인데 질문 문구가 수하물 지연을 다시 묻는 문제를 회귀 테스트로 고정한다.
- [x] target metadata와 질문 문구가 어긋날 때 target 슬롯 fallback 질문으로 보정한다.
- [x] 관련 단위 테스트와 전체 검증을 실행하고 의미 있는 단위로 커밋한다.
- [x] 이미 채워진 슬롯을 다시 묻는 질문을 남은 슬롯 질문으로 retarget하는 RED 테스트를 추가한다.
- [x] 기존 filled 슬롯 재질문 방지 보정을 구현한다.
- [x] retarget 회귀 테스트와 관련 테스트를 통과시키고 커밋한다.
- [x] `I am Trevor`, `SaudiStudy` 피드백 미화 방지 RED 테스트를 추가한다.
- [x] feedback prompt와 deterministic fallback을 grounding 기준으로 보강한다.
- [x] 피드백 회귀 테스트와 관련 테스트를 통과시키고 커밋한다.
- [x] 전체 테스트, compileall, diff check를 실행한다.

## newly filled slot 재질문 방지

- [x] dev 배포 후 세션 202 payload를 실제 AI API로 호출한다.
- [x] `next_options_request` 과잉 채움은 해결됐지만, 다음 질문이 방금 채운 `missed_connection`을 다시 묻는 문제를 확인한다.
- [x] 모델이 newly filled slot을 다시 묻는 케이스를 RED 테스트로 고정한다.
- [x] 남은 슬롯을 겨냥하지 않고 newly filled slot만 다시 묻는 질문을 남은 슬롯 질문으로 보정한다.
- [x] 관련 단위 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## request-like semantic evidence 과잉 채움 보정

- [x] 세션 202의 `next_options_request` 과잉 채움 케이스를 회귀 테스트로 고정한다.
- [x] 회귀 테스트가 현재 구현에서 실패하는지 RED를 확인한다.
- [x] 슬롯 description이 요청, 질문, 확인 행위를 요구할 때 최신 발화에 request act가 있는지 검증한다.
- [x] `next_options_request` 같은 요청형 슬롯은 상황 설명만으로 채워지지 않게 한다.
- [x] `Can you rebook me?`, `What should I do now?` 같은 실제 요청 발화는 계속 통과하게 한다.
- [x] next-question 프롬프트에 request-like 슬롯의 명시적 행위 기준을 반영한다.
- [x] 관련 단위 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## legacy slot evidence 제거

- [x] `evidencePolicy` 없는 슬롯 검증 fallback이 어디에 남아 있는지 확인한다.
- [x] `evidencePolicy` 없는 슬롯은 최종 `filledSlots`에 적용되지 않는 회귀 테스트를 추가한다.
- [x] `_legacy_slot_has_user_evidence()`와 슬롯명 switch fallback을 제거한다.
- [x] 기존 회귀 테스트를 typed `evidencePolicy` 기반으로 정리한다.
- [x] 단위 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## semantic evidence false negative 개선

- [x] BE dev 테스트 결과를 기준으로 실패 케이스를 AI 단위 테스트로 고정한다.
- [x] `My items came out too late.`가 raw `ASSISTANCE_REQUEST`여도 `baggage_delay_reason`을 채우도록 RED를 확인한다.
- [x] `My baggage came out too late.`와 `My baggage took too long.`도 같은 방식으로 RED를 확인한다.
- [x] `I missed my connecting flight.`는 `baggage_delay_reason`을 채우지 않는 기존 방어를 유지한다.
- [x] 한 문장 happy path에서 여러 semantic evidence 슬롯을 채우는 회귀 테스트를 추가한다.
- [x] `evidencePolicy` 기반 rescue pass를 슬롯명 switch 없이 구현한다.
- [x] semantic verifier 프롬프트를 핵심 evidence 기준으로 완화한다.
- [x] `ASSISTANCE_REQUEST` 최종화 조건을 실제 도움 요청일 때만 유지하도록 조정한다.
- [x] Obsidian semantic evidence 문서에 false negative 원인과 해결 결과를 기록한다.
- [x] 단위 테스트, 전체 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 의미 있는 단위로 커밋한다.

## semantic evidence 기반 슬롯 검증

- [x] 현재 next-question 요청 모델과 슬롯 검증 경로를 확인한다.
- [x] `EvidencePolicy` 요청 DTO를 추가하고 JSON object payload를 검증한다.
- [x] 세션 189의 `baggage_delay_reason` 과잉 채움 회귀 테스트를 추가하고 RED를 확인한다.
- [x] `my items came out too late`처럼 힌트에 없는 자유 표현이 semantic evidence로 통과하는 회귀 테스트를 추가한다.
- [x] `my items`처럼 의미 근거가 부족한 발화는 슬롯을 채우지 않는 회귀 테스트를 추가한다.
- [x] `candidateFilledSlots[].evidenceText` 기반 공통 검증을 구현한다.
- [x] next-question 프롬프트에 evidencePolicy와 evidenceText 기준을 반영한다.
- [x] 관련 단위 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 커밋한다.

## 관측성 로깅과 Sentry 추가

- [x] Sentry DSN은 SSM의 `/saynow/develop/SENTRY_DSN` 또는 `/saynow/prod/SENTRY_DSN`에 두는 것으로 정리한다.
- [x] 일반 로그가 Sentry breadcrumb로 붙도록 logging integration 설정을 명시하고 테스트한다.
- [x] 오류 로그와 `capture_exception`이 중복 이벤트를 만들지 않도록 logging event capture를 비활성화한다.
- [x] breadcrumb 설정과 SSM 적용 방식을 README와 context-notes에 기록한다.
- [x] 관련 단위 테스트, compileall, diff check를 다시 실행한다.
- [x] 현재 로깅, 설정, LLM 호출 구조를 확인한다.
- [x] Sentry DSN이 없으면 초기화하지 않는 설정 테스트를 추가한다.
- [x] Sentry DSN이 있으면 초기화 옵션이 적용되는 테스트를 추가한다.
- [x] API 생성 실패와 서버 내부 오류가 Sentry capture 경계로 전달되는 테스트를 추가한다.
- [x] next-question, feedback, guide 주요 단계의 소요 시간 로그 테스트를 추가한다.
- [x] Sentry 초기화와 예외 capture helper를 구현한다.
- [x] 오류가 날만한 LLM 호출, JSON 파싱, 응답 계약 검증 지점에 원인 추적 로그를 추가한다.
- [x] AI workflow 단계별 timing 로그를 확장한다.
- [x] 관련 단위 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 커밋한다.

## 동문서답 턴 분류 개선

- [x] `origin/develop` 기준 `feat/invalid-response-policy` 브랜치에서 작업한다.
- [x] 세션 159, 160의 문제 발화를 `next-question` 회귀 테스트로 고정한다.
- [x] 모델이 `INVALID_RESPONSE`를 반환하면 `filledSlots`를 무효화한다.
- [x] `ASSISTANCE_REQUEST`에서는 슬롯을 채우지 않도록 보정한다.
- [x] MVP 슬롯별 최소 증거 검증을 추가한다.
- [x] 역할 반전과 이미 채워진 슬롯 재질문 방지 테스트를 추가한다.
- [x] 세션 159, 160의 문제 발화를 `feedback` 회귀 테스트로 고정한다.
- [x] 문제 발화가 피드백에서 미화되지 않도록 보강한다.
- [x] RAG lookup, LLM chat, RAG save 단계별 소요 시간 로그를 추가한다.
- [x] Obsidian에 문제 해결 과정과 BE 후속 작업을 기록한다.
- [x] 단위 테스트, compileall, diff check를 실행한다.
- [x] 변경 사항을 커밋한다.

- [x] `main` 기준 `develop` 브랜치 생성.
- [x] 2차 MVP AI 서버 API 계약과 1차 MVP 제거 범위 확인.
- [x] `next-question`, `feedback` 새 계약 테스트를 먼저 추가하고 실패 확인.
- [x] 텍스트 기반 `conversation` 모델, 서비스, 라우터 구현.
- [x] 1차 MVP 오디오, STT, TTS API 등록과 obsolete 테스트 정리.
- [x] 전체 테스트와 diff 검증 실행.
- [x] 2차 MVP API 전환 커밋 생성.
- [x] develop EC2 배포용 GitHub Actions workflow 추가.
- [x] 발화별 피드백 문구 계약 회귀 테스트 추가.
- [x] `nativeUnderstanding`, `nativeLanguageInterpretation`, `betterExpression` 생성 기준 프롬프트 보강.
- [x] 배포 환경의 추가 `.env` 값 때문에 앱이 중단되지 않도록 설정 테스트와 방어 로직 추가.
- [x] 질문 의도와 다른 발화의 한국어 비유와 `betterExpression` 안내 형식 보강.
- [x] 피드백 응답 문자열 내부 큰따옴표와 역슬래시 사용 금지 조건 추가.
- [x] develop LLM provider를 Upstage로 전환할 수 있는 설정 추가.
- [x] `nativeLanguageInterpretation`의 비유 구간을 작은따옴표로 감싸도록 보강.
- [x] 무의미, 오프토픽, 거절 발화가 `next-question` 슬롯을 채우지 못하도록 방어.
- [x] 목표 실패 피드백 점수와 `betterExpression` 영어 우선 정책 보강.
- [x] `nativeLanguageInterpretation`이 프롬프트 예시 문장을 복사하지 않도록 회귀 테스트 추가.
- [x] 카페 옵션 발화의 한국어 비유를 같은 턴 `userUtterance` 의미로 보정.
- [x] 명시 검증한 오프토픽 발화의 한국어 비유가 literal 의미만 유지하도록 보정.
- [x] `nativeUnderstanding`이 `외국인은 ...고 이해했어요.` 형식만 사용하도록 회귀 테스트 추가.
- [x] 카페 옵션과 오프토픽 검증 발화의 `nativeUnderstanding`을 같은 턴 의미로 보정.
- [x] main 1차 MVP 피드백 기준 중 발화 품질, STT 비평 제외 항목, +1 개선 폭을 2차 MVP 프롬프트에 반영.
- [x] feedback 결과물의 deterministic validation, quality review, 1회 repair 루프 추가.
- [x] 좋은 응답 오판과 출력 형식 위반을 repair하는 회귀 테스트 추가.
- [x] repair 모델이 같은 문제를 반환해도 좋은 응답과 검증된 실패 발화는 코드 안전장치로 최종 보정.
- [x] reviewer가 통과시킨 좋은 응답 오판도 deterministic issue로 강제 보정.
- [x] `I don't know.` 피드백의 한국어 비유 중복 문구를 고정 문장으로 보정.
- [x] `I want`처럼 목적어가 빠진 미완성 발화의 피드백 회귀 테스트 추가.
- [x] 미완성 발화의 `nativeUnderstanding`, `nativeLanguageInterpretation` 프롬프트와 검증 규칙 보강.
- [x] 관련 단위 테스트와 diff 검증 실행.
- [x] `feedbackSummary`를 2문장 기본, 3문장 예외, 120자 이내로 제한하는 회귀 테스트 추가.
- [x] 긴 `feedbackSummary`를 repair 대상으로 보내는 deterministic 검증 추가.
- [x] 총평 길이 제한 관련 단위 테스트와 전체 검증 실행.
- [x] `next-question`에서 목적어 없는 주문 시작 발화가 슬롯을 채우지 않도록 회귀 테스트 추가.
- [x] `next-question`에서 `drink`, `something`, `menu` 같은 generic object 발화도 슬롯을 채우지 않도록 회귀 테스트 추가.
- [x] `feedback`의 미완성 주문 발화 보정을 `I want` 단일 케이스에서 공통 패턴으로 확장.
- [x] 미완성 주문 발화 공통 규칙의 단위 테스트와 live 배포 검증 실행.
- [x] `prompt-engineering-patterns` 기준으로 feedback 프롬프트를 classification, field policy, self-check 섹션으로 재구성.
- [x] 로컬 서버와 Dev 배포 서버에 동일 입력을 보내 프롬프트 개선 효과와 latency 비교.
- [x] `I want + 구체 음료`를 이해 가능하지만 +1 피드백이 필요한 near-miss로 고정.
- [x] near-miss 정책을 로컬 서버와 Dev 배포 서버에서 검증.
- [x] Solar Pro 3와 GPT-4o mini에 동일 피드백 입력을 넣고 출력 품질 비교.
- [x] Dev 배포 서버 LLM provider를 OpenAI GPT-4o mini로 전환하고 런타임 검증.
- [x] 시나리오 1 메뉴 추천 요청의 슬롯 추출 결과를 재현한다.
- [x] 시나리오 3 커스텀 음료 제작에서 `That’s all` 발화의 슬롯 추출 결과를 재현한다.
- [x] 재현 결과를 근거로 하트 차감 원인과 추가 검증 필요 여부를 정리한다.
- [x] Obsidian에 SayNow AI 프롬프트 실험 로그 문서 초안을 만든다.
- [x] 프롬프트별 기록 양식과 공통 10개 input을 문서 최상단에 고정한다.
- [x] vault 반영 경로와 파일 내용을 검증한다.
- [x] feedback 품질 테스트용 공통 10개 input과 기록 파트를 Obsidian 문서에 추가한다.
- [x] baseline 현재 프롬프트를 로컬 `conversation_service.py` 프롬프트 원문으로 교체한다.
- [x] NQ 기록 양식에 들어온 AI 질문, 사용자 입력, output 요약을 함께 남기도록 보강한다.
- [x] 현재 프롬프트로 `FB-01`부터 `FB-10`까지 실제 feedback 품질을 호출하고 Obsidian baseline에 결과를 채운다.
- [x] Prompt 2 방향인 프롬프트 정리, few-shot 보강, feedback judge 기준 보강의 회귀 테스트를 먼저 추가한다.
- [x] Prompt 2 프롬프트와 feedback review/repair 정책을 구현한다.
- [x] Prompt 2 테스트와 문서 반영을 검증한다.
- [x] Prompt 2를 로컬 실제 모델 호출로 `NQ-01`-`NQ-10`, `FB-01`-`FB-10` 재측정하고 Obsidian 결과를 채운다.
- [x] Prompt 3에서 카페 전용 판단을 도메인 중립 core prompt와 category example로 일반화한다.
- [x] 명확한 옵션/선호 답변을 좋은 응답으로 보는 정책을 회귀 테스트로 고정한다.
- [x] 공항, 호텔, 식당 smoke input을 추가해 로컬 실제 모델로 함께 검증한다.
- [x] `next-question` 응답 계약에 `turnClassification`을 추가하는 설계를 문서화한다.
- [x] 추천 요청, 정보 요청, 옵션 완료, 슬롯 답변, 실패 발화 분류 회귀 테스트를 추가한다.
- [x] `NextQuestionResponse` 모델과 route 응답 계약에 `turnClassification`을 반영한다.
- [x] LLM prompt와 deterministic fallback이 안정적인 분류를 반환하도록 구현한다.
- [x] 관련 단위 테스트와 로컬 실제 모델 smoke 검증을 실행한다.
- [x] `turnClassification`을 `ANSWER`, `ASSISTANCE_REQUEST`, `INVALID_RESPONSE` 3상태로 단순화하는 회귀 테스트를 추가한다.
- [x] 메뉴 정보 요청 응답에 실제 메뉴 항목이 보이도록 프롬프트와 보정 로직을 수정한다.
- [x] `availableOptions` 요청 계약을 추가해 메뉴와 옵션 정보를 structured context로 전달한다.
- [x] 메뉴와 추천 요청이 제공된 `availableOptions` 안에서만 응답하도록 회귀 테스트를 추가한다.
- [x] `availableOptions`가 없을 때 AI 서버가 구체 옵션을 지어내지 않도록 보정한다.
- [x] 관련 단위 테스트와 compileall, diff check를 실행한다.
- [x] `I need a menu`, `Can I get a menu`, `Menu please`가 `ASSISTANCE_REQUEST`로 처리되는 회귀 테스트를 추가한다.
- [x] `menu`를 generic object blocker에서 제거하고, 메뉴 요청은 도움 요청으로 판단하도록 프롬프트와 보정 로직을 수정한다.
- [x] Prompt 7 결과를 Obsidian 프롬프트 실험 로그에 기록한다.
- [x] 관련 단위 테스트와 compileall, diff check를 실행하고 커밋한다.
- [x] `prompt-engineering-patterns` 기준으로 next-question system prompt를 섹션화하는 회귀 테스트를 추가한다.
- [x] 메뉴와 추천 few-shot을 `availableOptions` 기반 예시로 바꾼다.
- [x] Prompt 8 결과를 Obsidian 프롬프트 실험 로그에 기록한다.
- [x] 관련 단위 테스트와 compileall, diff check를 실행하고 커밋한다.
- [x] Prompt 8 결과를 실제 OpenAI `gpt-4o-mini` live 호출로 재측정하고, 이후 프롬프트 실험 결과 표는 live 호출만 사용하도록 문서 기준을 정정한다.
- [x] `availableOptions`를 사용하지 않는 `ASSISTANCE_REQUEST` 전용 RAG workflow 방향을 Obsidian workflow 문서에 반영한다.
- [x] `availableOptions` 의존을 제거하고 `ASSISTANCE_REQUEST` 전용 RAG workflow 회귀 테스트를 추가한다.
- [x] pgvector 기반 도움 요청 저장소와 임베딩 호출 경계를 구현한다.
- [x] Prompt 9 live 테스트 결과를 Obsidian에 기록한다.
- [x] Supabase RAG 테이블 적용 SQL을 repo에 추가하고, live 검증에서 테이블 미생성 상태를 기록한다.
- [x] 반복 도움 요청을 자동으로 `candidate`로 승격하고 live DB에서 검증한다.
- [x] SSE 피드백 스트리밍용 summary/turn 단위 서비스 계약 추가.
- [x] `/api/v1/conversation/feedback/stream` SSE 라우터 추가.
- [x] SSE 이벤트 순서와 기존 동기 API 회귀 테스트 실행.
- [x] 피드백 요청에 백엔드 확정 `sessionResult` 계약 추가.
- [x] 기본 피드백과 SSE 피드백 프롬프트가 같은 `sessionResult`를 사용하도록 보강.
- [x] 관련 단위 테스트와 compileall, diff check를 실행하고 커밋한다.
- [x] `next-question`, `feedback`, `feedback/stream` 요청 계약에 `scenarioSituation` 필수 필드 추가.
- [x] 기본 피드백과 SSE 피드백 프롬프트가 같은 `scenarioSituation`을 사용하도록 테스트 먼저 추가.
- [x] 모델, 프롬프트, API 문서에 `scenarioSituation` 반영.
- [x] 관련 단위 테스트와 전체 검증 실행 후 커밋한다.
- [x] `humanizer` 기준으로 피드백 한국어의 공식형 표현을 줄이는 프롬프트 회귀 테스트 추가.
- [x] 동기 피드백, SSE summary, SSE turn feedback, repair 프롬프트에 `Natural Korean Style Policy` 반영.
- [x] `next-question`, `feedback`, `feedback/stream` 요청 계약에 `aiRole` 필수 필드 추가.
- [x] `aiRole`이 다음 질문, 동기 피드백, SSE 턴 피드백 프롬프트에 들어가도록 회귀 테스트 추가.
- [x] 모델, 프롬프트, API 문서에 `aiRole` 반영.
- [x] 관련 단위 테스트와 전체 검증 실행.
- [x] 커밋 후 `develop`으로 push한다.
- [x] 가이드 모드와 공통 방어 로직 설계를 문서화한다.
- [x] 프롬프트 인젝션 방어와 영어 학습 질문 허용 범위에 대한 실패 테스트를 먼저 추가한다.
- [x] 공통 safety guard를 추가하고 `next-question`, `feedback`, `feedback/stream`, `guide` 경계에 적용한다.
- [x] `/api/v1/conversation/guide` 요청/응답 모델과 라우트를 추가한다.
- [x] README에 가이드 모드 API 계약을 반영한다.
- [x] 관련 단위 테스트, compileall, diff check를 실행하고 커밋한다.
- [x] 가이드 API에서 직전 질문/발화 필드를 제거하고 extra field를 거부하도록 계약을 단순화한다.
- [x] 기획 심의 기술 꼬리 질문 범위를 현재 AI 서버 구현과 Obsidian 문서 기준으로 정리한다.
- [x] 프롬프트 품질 관리, 모델 선택, DB/인프라, 서버 분리, 할루시네이션 대응 Q&A를 Obsidian 노트로 작성한다.
- [x] 생성한 Obsidian 노트 경로와 주요 내용을 검증한다.
- [x] SSE 피드백에서 모든 턴이 good인데 summary만 교정성 문구를 내는 케이스를 회귀 테스트로 고정.
- [x] SSE summary에도 동기 feedback의 all-good 총평 보정을 적용.
- [x] 질문형 발화가 슬롯 설명상 확인 요청 자체를 수행하는 경우를 회귀 테스트로 고정.
- [x] `boarding_possibility`처럼 확인 요청형 슬롯은 assistance request로 오분류되어도 슬롯을 채우도록 보정.
- [x] 관련 단위 테스트와 배포 서버 품질 재검증 실행.
- [x] `slots[].description`을 `next-question` 슬롯 계약에 필수 필드로 추가.
- [x] `feedback`, `feedback/stream` 요청 계약에 `slots`와 `slots[].description` 필수 필드 추가.
- [x] 다음 질문, 동기 피드백, SSE summary/turn feedback 프롬프트가 슬롯 설명을 사용하도록 반영.
- [x] API 문서와 관련 테스트 payload를 새 슬롯 계약에 맞게 갱신.
- [x] 관련 단위 테스트와 전체 검증 실행.
- [x] 커밋 후 `develop`으로 push한다.
