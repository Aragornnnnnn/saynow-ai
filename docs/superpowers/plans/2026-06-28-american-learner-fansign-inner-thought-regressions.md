# American Learner Fansign Inner Thought Regressions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the AMERICAN_LEARNER fan-sign innerThoughtType regressions shown in the user-provided live session log.

**Architecture:** Keep the existing prompt and repair pipeline. Add representative RED tests, then extend the rubric-based fallback issue classifier with reusable intent/tone patterns instead of full sentence matching.

**Tech Stack:** Python 3.12, unittest, existing SayNow AI conversation service.

## Global Constraints

- Scope is `saynow-ai` only.
- Do not change FE/BE API contracts.
- Do not change turn-feedback/nativeScore behavior.
- Do not add exact session-id or full-row hardcoding.
- Keep `AMERICAN_LEARNER` innerThought in English.
- Treat short direct greeting answers as GOOD when they satisfy the greeting.
- Treat song-question answers that give no song and no reason as BAD.
- Treat direct personal criticism in a fan-sign closing as BAD.

---

### Task 1: Add RED Regression Tests

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `generate_next_question`, `generate_closing_message`.
- Produces: failing tests for the three live regressions.

- [x] **Step 1: Add a GOOD greeting regression.**

`잘 지냈어` after `잘 지냈어?` must become `innerThoughtType=GOOD` even if the model returns `NORMAL`.

- [x] **Step 2: Add a BAD song-intent regression.**

`나 아이돌` after `우리 노래 중에는 뭐가 제일 좋아? 왜 그 노래야?` must become `BAD`.

- [x] **Step 3: Add a BAD direct-critique closing regression.**

`너 너무 기계적이다.` in the final fan-sign prompt must become `BAD`, and a softened `appreciate the honesty` thought must be replaced.

- [x] **Step 4: Verify RED.**

Run focused tests and confirm failures come from the missing rubric handling.

### Task 2: Implement Minimal Rubric Repairs

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/app/services/conversation_service.py`

**Interfaces:**
- Consumes: existing `_conversation_issue_kind`, `_fallback_inner_thought_type`, `_fallback_inner_thought_en`.
- Produces: reusable greeting, missing-song-intent, and personal-critique issue detection.

- [x] **Step 1: Add greeting GOOD detection.**

Before short-answer downgrade, detect Korean direct answers to greeting/well-being questions.

- [x] **Step 2: Add song/reason missing-intent detection.**

Detect favorite-song questions where the answer only gives a self/role label or generic idol/fan identity instead of song/reason content.

- [x] **Step 3: Add direct personal critique detection.**

Detect Korean/English direct criticism like mechanical/robotic/bot-like comments as relationship-damaging.

- [x] **Step 4: Verify GREEN.**

Focused tests must pass.

### Task 3: Full Verification And Commit

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/checklist.md`
- Modify: `/Users/sangmin8817/Soma/saynow-ai/context-notes.md`

**Interfaces:**
- Consumes: verification outputs.
- Produces: documented evidence and one commit.

- [x] **Step 1: Run checks.**

Run focused tests, `tests.test_conversation_service`, full unittest discover, compileall, and `git diff --check`.

- [x] **Step 2: Update notes and commit.**

Record RED/GREEN evidence and commit the logical fix.
