# Available Options Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured `availableOptions` input to the next-question workflow so the AI server can answer menu, recommendation, and option-help requests without inventing choices.

**Architecture:** Keep next-question as a latency-critical single-model workflow. The LLM still runs once, but request context now includes available options, and deterministic normalization rewrites empty or unsupported assistance answers into responses grounded in the provided options.

**Tech Stack:** FastAPI, Pydantic v2, Python `unittest`, existing `conversation_service` prompt and normalization helpers.

---

### Task 1: Add `availableOptions` Request Contract

**Files:**
- Modify: `app/models/conversation.py`
- Test: `tests/test_conversation_service.py`

- [x] **Step 1: Write failing model/service tests**

Add tests showing `NextQuestionRequest` accepts optional `availableOptions`, rejects blank option values, and keeps existing requests valid when the field is omitted.

- [x] **Step 2: Run the focused tests to verify RED**

Run: `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_next_question_accepts_available_options_context`

Expected: fail because `availableOptions` is not modeled yet.

- [x] **Step 3: Implement the minimal request model**

Add `AvailableOptionRequest` with `slotName` and `options`, then add `availableOptions: list[AvailableOptionRequest] = Field(default_factory=list)` to `NextQuestionRequest`.

- [x] **Step 4: Run the focused tests to verify GREEN**

Run the same focused test and confirm it passes.

### Task 2: Ground Assistance Responses in Available Options

**Files:**
- Modify: `app/services/conversation_service.py`
- Test: `tests/test_conversation_service.py`

- [x] **Step 1: Write failing behavior tests**

Add tests for these cases.

- Menu request with `availableOptions` returns visible options from the request.
- Recommendation request with `availableOptions` recommends an allowed option if the model suggests an unavailable one.
- Information request without `availableOptions` does not invent menu items.

- [x] **Step 2: Run focused tests to verify RED**

Run the new focused tests and confirm failures are about missing option grounding.

- [x] **Step 3: Implement option-aware normalization**

Extend the existing visible information response helper so it picks options from the first unfilled slot with available options, rewrites generic menu text into an option-grounded answer, and avoids fabricated options when no options are provided.

- [x] **Step 4: Run focused tests to verify GREEN**

Run the focused tests and confirm they pass.

### Task 3: Prompt and Documentation Sync

**Files:**
- Modify: `app/services/conversation_service.py`
- Modify: `docs/superpowers/specs/2026-05-24-next-question-turn-classification-design.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [x] **Step 1: Prompt test first**

Assert that the next-question prompt tells the model to use only `availableOptions` when provided and not invent options outside the provided list.

- [x] **Step 2: Update prompt and user prompt context**

Add an `Available options for unfilled slots` section to the user prompt and add system prompt instructions that options must be grounded in that section.

- [x] **Step 3: Update docs and context notes**

Record the workflow decision that this is structured context, not RAG, and that `availableOptions` is the source of truth for option/menu content.

### Task 4: Verification and Commit

**Files:**
- All touched files

- [x] **Step 1: Run full tests**

Run: `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest discover -s tests -p 'test*.py'`

- [x] **Step 2: Run compile check**

Run: `/private/tmp/saynow-ai-venv/bin/python -m compileall app tests`

- [x] **Step 3: Run diff check**

Run: `git diff --check`

- [x] **Step 4: Commit**

Commit message: `feat: availableOptions 기반 꼬리 질문 보강`
