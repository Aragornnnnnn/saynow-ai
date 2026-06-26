# US Market Korean Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** BE가 AI 서버 요청에 서비스 대상을 전달하면 기존 한국인 대상 영어 회화와 신규 미국인 대상 한국어 회화 프롬프트를 분기한다.

**Architecture:** `serviceAudience`를 요청 계약에 추가하고 기본값은 기존 `KOREAN_LEARNER`로 둔다. `AMERICAN_LEARNER` 요청에서는 프롬프트를 미국인 한국어 학습자 기준으로 바꾸고, GOOD 턴의 `benchmarkMessage`를 서버 후처리에서 `null`로 강제한다.

**Tech Stack:** FastAPI, Pydantic v2, Python `unittest`, process-memory turn feedback cache.

## Global Constraints

- FE-BE API 구조는 유지하고 BE-AI 요청에만 대상 구분값을 추가한다.
- 기존 요청은 `serviceAudience`를 생략해도 한국인 대상 영어 회화 모드로 동작해야 한다.
- 미국인 대상 한국어 회화 모드의 `benchmarkMessage`는 `null`로 내려간다.
- 기존 한국인 대상 영어 회화 모드의 `benchmarkMessage` 정책은 유지한다.
- 새 source file은 만들지 않는다.

---

### Task 1: 요청 계약에 서비스 대상 추가

**Files:**
- Modify: `app/models/conversation.py`
- Test: `tests/test_conversation_service.py`

**Interfaces:**
- Produces: `ServiceAudience.KOREAN_LEARNER`, `ServiceAudience.AMERICAN_LEARNER`
- Produces: `ScenarioContext.serviceAudience`
- Produces: `GuideChatRequest.serviceAudience`

- [ ] **Step 1: Write failing tests**

Add tests that validate omitted `serviceAudience` defaults to `KOREAN_LEARNER`, scenario requests accept `AMERICAN_LEARNER`, and guide requests accept `AMERICAN_LEARNER`.

- [ ] **Step 2: Run focused tests**

Run: `OPENAI_API_KEY=test-key .venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_service_audience_defaults_to_korean_learner tests.test_conversation_service.ConversationServiceTest.test_service_audience_accepts_american_learner_for_scenario_requests tests.test_conversation_service.ConversationServiceTest.test_guide_request_accepts_american_learner -v`

Expected: FAIL because the enum and fields do not exist.

- [ ] **Step 3: Implement minimal model changes**

Add the enum and defaulted fields in `app/models/conversation.py`.

- [ ] **Step 4: Re-run focused tests**

Expected: PASS.

### Task 2: 대상별 프롬프트 분기 추가

**Files:**
- Modify: `app/services/conversation_service.py`
- Test: `tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `request.scenario.serviceAudience`
- Consumes: `request.serviceAudience`

- [ ] **Step 1: Write failing tests**

Add tests that capture prompts for next-question, turn-feedback, session-feedback, and guide in `AMERICAN_LEARNER` mode.

- [ ] **Step 2: Run focused tests**

Run: `OPENAI_API_KEY=test-key .venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_american_learner_next_question_prompt_targets_korean_conversation tests.test_conversation_service.ConversationServiceTest.test_american_learner_closing_message_prompt_targets_korean_conversation tests.test_conversation_service.ConversationServiceTest.test_american_learner_session_feedback_prompt_targets_korean_learning -v`

Expected: FAIL because prompts still describe Korean learners practicing English.

- [ ] **Step 3: Implement minimal prompt helpers**

Add small helper functions that return audience-specific labels and instructions without restructuring the large service file.

- [ ] **Step 4: Re-run focused tests**

Expected: PASS.

### Task 3: 미국인 대상 benchmarkMessage null 정책 적용

**Files:**
- Modify: `app/services/conversation_service.py`
- Test: `tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `TurnFeedbackRequest.scenario.serviceAudience`
- Produces: cached `TurnFeedbackData.benchmarkMessage is None` for `AMERICAN_LEARNER`

- [ ] **Step 1: Write failing tests**

Add a GOOD turn test where the LLM returns a benchmark and detected patterns in `AMERICAN_LEARNER` mode, then assert the cached feedback has `benchmarkMessage is None`.

- [ ] **Step 2: Run focused test**

Run: `OPENAI_API_KEY=test-key .venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_american_learner_good_turn_feedback_forces_benchmark_message_null -v`

Expected: FAIL because the existing postprocessor fills a Korean-learner benchmark.

- [ ] **Step 3: Implement minimal postprocess branch**

Return a validated copy with `benchmarkMessage=None` before Korean benchmark generation when the request is `AMERICAN_LEARNER`.

- [ ] **Step 4: Re-run focused test**

Expected: PASS.

### Task 4: Guide safety and docs verification

**Files:**
- Modify: `app/services/safety_guard.py`
- Modify: `app/services/conversation_service.py`
- Modify: `readme.md`
- Test: `tests/test_conversation_service.py`

**Interfaces:**
- Consumes: guide `serviceAudience`
- Produces: American learner guide questions about Korean are allowed.

- [ ] **Step 1: Write failing guide safety test**

Add a guide test where an American learner asks a Korean expression question and assert the model is called instead of returning the English-only blocked answer.

- [ ] **Step 2: Run focused test**

Run: `OPENAI_API_KEY=test-key .venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_american_learner_guide_allows_korean_learning_questions -v`

Expected: FAIL because guide safety currently checks only English-learning markers.

- [ ] **Step 3: Implement minimal safety branch**

Let guide safety accept English-learning or Korean-learning markers based on `serviceAudience`, while keeping prompt-injection blocking shared.

- [ ] **Step 4: Run verification**

Run: `OPENAI_API_KEY=test-key .venv/bin/python -m unittest discover -s tests -p 'test*.py'`

Run: `.venv/bin/python -m compileall app tests`

Run: `git diff --check`

Expected: all pass.
