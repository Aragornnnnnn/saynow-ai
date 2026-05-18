# 작업 맥락 기록

- `main`은 1차 MVP 운영 코드이고, `develop`은 2차 MVP 개발 브랜치다.
- 2차 MVP AI 서버는 백엔드가 호출하는 내부 API만 제공한다. 인증은 애플리케이션 코드가 아니라 AWS Security Group 경계에서 처리한다.
- 1차 MVP의 오디오 업로드, Whisper STT, OpenAI TTS, `/api/v1/turn-evaluations`, `/api/v1/session-feedbacks`는 하위 호환 없이 제거해도 된다.
- 새 `POST /api/v1/conversation/next-question`은 텍스트 `userUtterance`와 현재 슬롯 상태를 받아 다음 꼬리 질문을 생성한다.
- `next-question`의 `filledSlots`는 이번 발화로 새롭게 충족된 슬롯만 반환한다. 이미 `filled=true`로 들어온 슬롯은 반환하지 않는다.
- 백엔드가 보낸 미충족 슬롯이 모두 이번 발화로 채워졌다고 판단되면 `nextQuestion`과 `translatedQuestion`은 `null`로 반환한다.
- 세션 완료 여부와 누적 슬롯 상태 저장은 백엔드 책임이다. AI 서버는 새로 충족된 슬롯과 다음 질문 후보만 반환한다.
- 새 `POST /api/v1/conversation/feedback`은 완료된 세션의 텍스트 턴 목록을 받아 전체 이해도, 총평, 턴별 피드백을 생성한다.
- 1차 MVP 코드 정리는 앱 라우터 등록 제거에 그치지 않고, 오디오/STT/TTS/로컬 시나리오 기반 라우터, 모델, 서비스, 테스트 파일 삭제까지 포함했다.
- 검증 명령은 `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest discover -s tests -p 'test*.py'`와 `git diff --check`를 실행했다.
