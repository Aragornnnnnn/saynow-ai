# Score, Inner Thought, Correction Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make score mix policy audience-neutral, make AMERICAN_LEARNER innerThought sound like a human private reaction, and keep Korean correctionExpression anchored to the user's original intent.

**Architecture:** Keep language-specific scoring for English versus Korean utterance length and complexity, but move feedback-mix score caps out of the AMERICAN_LEARNER-only branch. Tune prompts and narrow deterministic fallbacks without adding new broad fallback behavior.

**Tech Stack:** Python, Pydantic DTOs, unittest, existing `conversation_service.py` prompt and postprocess helpers.

## Global Constraints

- Do not change the FE-BE API shape.
- Keep `AMERICAN_LEARNER` `benchmarkMessage` as `null`.
- Keep `AMERICAN_LEARNER` visible Korean practice fields in Korean and explanations for the learner in English.
- Preserve the user's original intent when generating `correctionExpression`; do not invent new preferences.
- Use TDD for behavior changes.

---

### Task 1: Common Feedback-Mix Score Policy

**Files:**
- Modify: `app/services/conversation_service.py`
- Test: `tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `_postprocess_native_score_for_feedback_mix(native_score, turn_feedback_entries, service_audience)`
- Produces: A feedback-mix cap that applies to both `KOREAN_LEARNER` and `AMERICAN_LEARNER`.

- [x] Write a failing Korean-learner test where all cached turns are `NEEDS_IMPROVEMENT` but the raw score would be above the good band.
- [x] Run the focused test and verify it fails because the Korean learner score is not capped.
- [x] Remove the audience guard from `_postprocess_native_score_for_feedback_mix` while keeping language-specific score components unchanged.
- [x] Run the focused score tests and verify they pass.

### Task 2: Human AMERICAN_LEARNER Inner Thought Tone

**Files:**
- Modify: `app/services/conversation_service.py`
- Test: `tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `_american_learner_next_question_system_prompt()`, `_american_learner_closing_message_system_prompt()`, `_fallback_inner_thought()`
- Produces: Prompt and fallback guidance that avoids report-style `They seem... which makes...` innerThought.

- [x] Write a failing prompt test requiring short, immediate, non-report-style innerThought guidance.
- [x] Run the focused test and verify it fails on missing tone constraints.
- [x] Add concise guidance against observer-report phrasing and long causal summaries.
- [x] Update the remaining AMERICAN_LEARNER fallback innerThought that currently starts with `They`.
- [x] Run focused innerThought tests and verify they pass.

### Task 3: Preserve User Intent In Korean Correction Expression

**Files:**
- Modify: `app/services/conversation_service.py`
- Test: `tests/test_conversation_service.py`

**Interfaces:**
- Consumes: AMERICAN_LEARNER turn feedback prompt and `_needs_feedback_for_missing_required_question_intent()`
- Produces: Correction expressions that minimally improve the submitted Korean without inventing a new food preference.

- [x] Write a failing prompt test comparing AMERICAN_LEARNER to KOREAN_LEARNER intent-preservation guidance.
- [x] Write a failing deterministic test for `아무거나요. 상관없어요.` expecting a preserved-intent correction.
- [x] Run focused tests and verify both fail.
- [x] Add explicit `do not invent new preference` guidance to AMERICAN_LEARNER field policy.
- [x] Replace blind-date food examples and deterministic correction with an intent-preserving expression.
- [x] Run focused turn-feedback tests and verify they pass.

### Task 4: Verification And Commit

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: All changed tests and service code.
- Produces: Verified commit-ready working tree.

- [x] Run focused score, innerThought, and correction tests.
- [x] Run full `tests.test_conversation_service`.
- [x] Run full unittest discovery.
- [x] Run `compileall app tests scripts`.
- [x] Run `git diff --check`.
- [x] Update checklist and context notes with results.
- [x] Commit one logical change.
