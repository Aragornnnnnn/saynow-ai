# Feedback Scenario Result Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend-confirmed scenario success or failure to both sync and SSE feedback request handling.

**Architecture:** `ConversationFeedbackRequest` receives `scenarioResult` as a required enum. The sync feedback prompt, summary prompt, and turn prompt include the backend-confirmed result so generated feedback stays consistent with session outcome. The existing sync and SSE endpoints continue sharing the same request model.

**Tech Stack:** FastAPI, Pydantic, Python unittest.

---

### Task 1: Request Contract

**Files:**
- Modify: `app/models/conversation.py`
- Test: `tests/test_conversation_service.py`

- [x] **Step 1: Write failing tests**

Add tests that validate `scenarioResult` accepts only `SUCCESS` or `FAILURE` and that feedback prompts include it.

- [x] **Step 2: Run focused tests**

Run: `/private/tmp/saynow-ai-venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest`

Expected: fails because `scenarioResult` is not modeled or included in prompts.

- [x] **Step 3: Implement request enum and prompt inclusion**

Add `ScenarioResult` enum to `app/models/conversation.py`, add `scenarioResult` to `ConversationFeedbackRequest`, and include it in feedback user prompts.

- [x] **Step 4: Run focused tests again**

Run: `/private/tmp/saynow-ai-venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest`

Expected: pass.

### Task 2: Route Contract

**Files:**
- Modify: `tests/test_conversation_routes.py`

- [x] **Step 1: Update route request examples**

Add `scenarioResult` to sync and SSE feedback route test payloads.

- [x] **Step 2: Run route tests**

Run: `/private/tmp/saynow-ai-venv/bin/python -m unittest tests.test_conversation_routes.ConversationRoutesTest`

Expected: pass.

### Task 3: Verification

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [x] **Step 1: Run all tests**

Run: `/private/tmp/saynow-ai-venv/bin/python -m unittest discover -s tests -p 'test*.py'`

Expected: pass.

- [x] **Step 2: Run compile and diff checks**

Run: `/private/tmp/saynow-ai-venv/bin/python -m compileall app tests`

Expected: pass.

Run: `git diff --check`

Expected: pass.

- [x] **Step 3: Commit**

Run: `git add app/models/conversation.py app/services/conversation_service.py tests/test_conversation_routes.py tests/test_conversation_service.py checklist.md context-notes.md docs/superpowers/plans/2026-05-24-feedback-scenario-result.md`

Run: `git commit -m "sse-feedback-stream feat: 피드백 요청 시나리오 결과 반영"`
