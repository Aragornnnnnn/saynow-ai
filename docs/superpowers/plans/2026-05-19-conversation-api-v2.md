# Conversation API V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 1st MVP audio/STT/TTS AI API with the 2nd MVP text-only conversation APIs consumed by the backend.

**Architecture:** Keep FastAPI as the HTTP boundary and add a single `conversation` route module. Put request/response DTOs in `app/models/conversation.py` and LLM orchestration in `app/services/conversation_service.py`. Remove old router registration and delete obsolete 1st MVP tests so the registered API surface matches the new backend-facing contract.

**Tech Stack:** FastAPI, Pydantic v2, OpenAI chat client wrapper, Python `unittest`.

---

### Task 1: Planning Artifacts

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [ ] **Step 1: Replace checklist with 2nd MVP tasks.**

Expected checklist items:
- branch setup
- RED tests
- conversation DTO/service/router implementation
- old API removal
- verification
- commit

- [ ] **Step 2: Replace context notes with confirmed decisions.**

Required notes:
- `develop` is the 2nd MVP branch from `main`.
- Old audio/STT/TTS APIs can be removed.
- New APIs are text-only.
- `filledSlots` returns only newly satisfied slots.
- If all currently unfilled slots are newly satisfied, `nextQuestion` and `translatedQuestion` are `null`.

### Task 2: RED Tests For New Contract

**Files:**
- Create: `tests/test_conversation_service.py`
- Create: `tests/test_conversation_routes.py`

- [ ] **Step 1: Add service tests for `next-question`.**

Test behaviors:
- only newly satisfied unfilled slots are returned
- already filled slots are never returned
- all unfilled slots satisfied forces `nextQuestion=None` and `translatedQuestion=None`

- [ ] **Step 2: Add service tests for `feedback`.**

Test behaviors:
- response preserves backend `turnId`
- response exposes `feedbackRequired`, `nativeUnderstanding`, `nativeLanguageInterpretation`, `betterExpression`
- invalid AI JSON raises a generation error

- [ ] **Step 3: Add route tests for public API surface.**

Test behaviors:
- `POST /api/v1/conversation/next-question` returns the documented JSON shape
- `POST /api/v1/conversation/feedback` returns the documented JSON shape
- old endpoints such as `/api/v1/turn-evaluations` are no longer registered

- [ ] **Step 4: Run RED tests.**

Run: `OPENAI_API_KEY=test-key python -m unittest tests.test_conversation_service tests.test_conversation_routes`

Expected: fail because `app.models.conversation`, `app.services.conversation_service`, and `app.api.routes.conversation` do not exist yet.

### Task 3: Implement Text-Only Conversation API

**Files:**
- Create: `app/models/conversation.py`
- Create: `app/services/conversation_service.py`
- Create: `app/api/routes/conversation.py`
- Modify: `app/main.py`

- [ ] **Step 1: Add Pydantic DTOs.**

DTOs:
- `NextQuestionRequest`
- `SlotStatusRequest`
- `NextQuestionResponse`
- `FilledSlotResponse`
- `ConversationFeedbackRequest`
- `FeedbackTurnRequest`
- `ConversationFeedbackResponse`
- `TurnFeedbackResponse`

- [ ] **Step 2: Implement service functions.**

Functions:
- `generate_next_question(request)`
- `generate_feedback(request)`

Both functions parse LLM JSON strictly and raise `ConversationGenerationError` on invalid or incomplete model output.

- [ ] **Step 3: Implement FastAPI routes.**

Routes:
- `POST /api/v1/conversation/next-question`
- `POST /api/v1/conversation/feedback`

Validation errors should return HTTP 400 with `{"code": "INVALID_REQUEST", "message": "잘못된 요청입니다."}`.

- [ ] **Step 4: Register only the new conversation router in `app/main.py`.**

Keep `/health`.

### Task 4: Remove 1st MVP API Surface

**Files:**
- Delete: `app/api/routes/scenario.py`
- Delete: `app/api/routes/session_feedback.py`
- Delete: `app/api/routes/stt.py`
- Delete: `app/api/routes/tts.py`
- Delete: `app/api/routes/turn_evaluation.py`
- Delete: `app/models/scenario.py`
- Delete: `app/models/session_feedback.py`
- Delete: `app/models/turn_evaluation.py`
- Delete: `app/services/scenario_service.py`
- Delete: `app/services/session_feedback_service.py`
- Delete: `app/services/stt_service.py`
- Delete: `app/services/tts_service.py`
- Delete: `app/services/turn_evaluation_service.py`
- Delete: `app/data/scenarios.json`
- Delete: `tests/test_turn_evaluation_service.py`
- Delete: `tests/test_session_feedback_service.py`
- Delete: `tests/test_stt_service.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Remove obsolete tests for deleted behavior.**

The new contract has no audio upload, STT, or TTS behavior.

- [ ] **Step 2: Remove `python-multipart` from runtime dependencies.**

No endpoint accepts multipart form data after the 2nd MVP conversion.

### Task 5: Verification And Commit

**Files:**
- All modified files

- [ ] **Step 1: Run focused tests.**

Run: `OPENAI_API_KEY=test-key python -m unittest tests.test_conversation_service tests.test_conversation_routes`

Expected: pass.

- [ ] **Step 2: Run full tests.**

Run: `OPENAI_API_KEY=test-key python -m unittest discover -s tests -p 'test*.py'`

Expected: pass.

- [ ] **Step 3: Inspect diff.**

Run: `git diff --stat` and `git diff --check`.

Expected: no whitespace errors and diff limited to 2nd MVP API conversion.

- [ ] **Step 4: Commit one logical change.**

Commit message: `feat: 2차 MVP 대화 API로 전환`
