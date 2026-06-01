# Safety Guard And Guide Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guide mode for English-learning questions and apply shared prompt-injection defense to every LLM-facing conversation path.

**Architecture:** Create a small safety guard service with purpose-specific checks. Keep API responses inside existing contracts, add a separate `/api/v1/conversation/guide` route, and include a shared safety policy in LLM system prompts.

**Tech Stack:** FastAPI, Pydantic, unittest, OpenAI-compatible chat client.

---

### Task 1: Safety Guard Tests

**Files:**
- Modify: `tests/test_conversation_service.py`

- [x] Add tests proving prompt-injection input is blocked before `next-question` calls the LLM.
- [x] Add tests proving guide mode blocks injection and off-topic questions without calling the LLM.
- [x] Add tests proving guide mode allows an English usage question and validates a JSON answer.
- [x] Run the focused tests and confirm they fail before implementation.

### Task 2: Safety Guard Implementation

**Files:**
- Create: `app/services/safety_guard.py`
- Modify: `app/services/conversation_service.py`
- Modify: `app/models/conversation.py`

- [x] Add purpose-specific safety checks.
- [x] Add guide request and response models.
- [x] Apply the guard to `generate_next_question`.
- [x] Add shared safety policy text to next-question, feedback, feedback-summary, turn-feedback, repair, and guide prompts.
- [x] Implement `generate_guide_answer`.
- [x] Run focused service tests and make them pass.

### Task 3: Route And Docs

**Files:**
- Modify: `app/api/routes/conversation.py`
- Modify: `tests/test_conversation_routes.py`
- Modify: `README.md`

- [x] Add route test for `/api/v1/conversation/guide`.
- [x] Add the guide route.
- [x] Document the request and response contract.
- [x] Run route tests.

### Task 4: Verification And Commit

- [x] Run `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest discover -s tests -p 'test*.py'`.
- [x] Run `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m compileall app tests`.
- [x] Run `git diff --check`.
- [x] Commit the logical change on `feat/16`.
