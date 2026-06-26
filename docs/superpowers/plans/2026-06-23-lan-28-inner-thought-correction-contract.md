# LAN-28 Inner Thought And Correction Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add role-aware `innerThought` output to next-question and split turn-feedback correction text into `correctionExpression` and `correctionReason`.

**Architecture:** Keep the existing FastAPI route surface and Pydantic DTO layer. Add the new response fields to `NextQuestionResponse`, require `scenario.counterpartRole`, and update next-question prompt, fallback, and repair paths so every response carries an inner thought. Change `TurnFeedbackData` so `NEEDS_IMPROVEMENT` carries separate correction expression and reason while GOOD keeps `feedbackDetail` and `benchmarkMessage`.

**Tech Stack:** FastAPI, Pydantic v2, Python unittest, OpenAI-compatible chat wrapper.

---

### Task 1: next-question 속마음 계약

**Files:**
- Modify: `tests/test_conversation_service.py`
- Modify: `tests/test_conversation_routes.py`
- Modify: `app/models/conversation.py`
- Modify: `app/services/conversation_service.py`

- [x] Add RED tests requiring `scenario.counterpartRole`, `innerThought`, and `innerThoughtType`.
- [x] Run focused RED tests for the next-question route and service contract.
- [x] Update DTO, prompt, fallback, and drift repair paths.
- [x] Re-run the focused tests and keep existing next-question tests green.

### Task 2: turn-feedback 개선 표현 계약

**Files:**
- Modify: `tests/test_conversation_service.py`
- Modify: `tests/test_conversation_routes.py`
- Modify: `app/models/conversation.py`
- Modify: `app/services/conversation_service.py`

- [x] Add RED tests requiring `correctionExpression` and `correctionReason` for `NEEDS_IMPROVEMENT`.
- [x] Run focused RED tests for the turn-feedback contract.
- [x] Update DTO validation, prompt schema, legacy normalization, deterministic repairs, and session aggregation.
- [x] Re-run focused tests.

### Task 3: 전체 검증

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [x] Run `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-lan28-venv/bin/python -m unittest tests.test_conversation_service tests.test_conversation_routes`.
- [x] Run `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-lan28-venv/bin/python -m unittest discover -s tests -p 'test*.py'`.
- [x] Run `/private/tmp/saynow-ai-lan28-venv/bin/python -m compileall app tests`.
- [x] Run `git diff --check`.
- [x] Commit with message `LAN-28 feat: 속마음과 개선 표현 계약 반영`.
