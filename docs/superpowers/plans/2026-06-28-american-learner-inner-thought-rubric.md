# American Learner Inner Thought Rubric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved GOOD/NORMAL/BAD innerThoughtType rubric for AMERICAN_LEARNER conversation outputs.

**Architecture:** Keep the external API shape unchanged. Update only the AMERICAN_LEARNER conversation prompts and the narrow fallback/repair classifier that assigns innerThoughtType when model output is invalid, generic, or inconsistent with the rubric.

**Tech Stack:** Python 3.12, FastAPI service functions, Pydantic models, unittest.

## Global Constraints

- Scope is `saynow-ai` only.
- Do not change FE/BE API contracts.
- Do not add hardcoded full-utterance classifiers for one screenshot sentence.
- `GOOD` means the answer satisfies the question or situation's core intent, is clear, and is acceptable for the counterpart role.
- `NORMAL` means the core intent is mostly satisfied, but information or relationship tone is weak enough to feel slightly incomplete.
- `BAD` means the core intent is not satisfied, meaning is hard to understand, or the counterpart would feel confused, hurt, distant, or uncomfortable.
- Verify with RED/GREEN tests before implementation completion.

---

### Task 1: Lock The Rubric Into Tests

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `ConversationService._next_question_system_prompt`, `_closing_message_system_prompt`, `generate_next_question`
- Produces: failing tests that describe the accepted rubric.

- [ ] **Step 1: Write the failing prompt test.**

Add assertions that AMERICAN_LEARNER next-question and closing-message prompts include the approved definitions for GOOD, NORMAL, and BAD.

- [ ] **Step 2: Write the failing behavior test.**

Add a case where the question asks for a bias but the user only says `응 좋아해.`. Even if the model returns `NORMAL`, final `innerThoughtType` must be `BAD` because the core intent is not answered.

- [ ] **Step 3: Write the guard behavior tests.**

Add cases where `민지.` stays `NORMAL` and `영상 봤어.` stays `NORMAL` because the core intent is mostly present but thin.

- [ ] **Step 4: Run focused tests and verify RED.**

Run:

```bash
.venv/bin/python -m unittest \
  tests.test_conversation_service.ConversationServiceTest.test_american_learner_conversation_prompts_include_approved_inner_thought_type_rubric \
  tests.test_conversation_service.ConversationServiceTest.test_american_learner_bias_question_generic_like_answer_is_bad_inner_thought \
  tests.test_conversation_service.ConversationServiceTest.test_american_learner_thin_core_intent_answers_stay_normal_inner_thought
```

Expected: at least the new prompt test and missing-intent behavior test fail before implementation.

### Task 2: Apply Prompt And Fallback Rubric

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/app/services/conversation_service.py`

**Interfaces:**
- Consumes: existing `ServiceAudience`, `NextQuestionRequest`, `ClosingMessageRequest`, and repair flow.
- Produces: AMERICAN_LEARNER prompts and fallback type classifier that follow the approved rubric.

- [ ] **Step 1: Update AMERICAN_LEARNER prompt wording.**

Replace the broad `clear, warm, or appropriate` wording with the approved definitions and add a short self-check for core intent, clarity, and relationship fit.

- [ ] **Step 2: Add narrow core-intent issue detection.**

Add helper logic for AMERICAN_LEARNER fan scenario questions where a required slot is missing, starting with bias questions that get only generic agreement such as `응 좋아해.`.

- [ ] **Step 3: Keep thin-but-relevant answers as NORMAL.**

Do not classify short but on-topic answers like `민지.` or `영상 봤어.` as BAD.

- [ ] **Step 4: Run focused tests and verify GREEN.**

Run the same focused command from Task 1. Expected: PASS.

### Task 3: Verify And Commit

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/checklist.md`
- Modify: `/Users/sangmin8817/Soma/saynow-ai/context-notes.md`

**Interfaces:**
- Consumes: test outputs from Task 2.
- Produces: documented verification evidence and one logical commit.

- [ ] **Step 1: Run relevant tests.**

Run:

```bash
.venv/bin/python -m unittest tests.test_conversation_service
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
.venv/bin/python -m compileall app tests scripts
git diff --check
```

Expected: all pass.

- [ ] **Step 2: Update notes.**

Record the rubric, RED/GREEN result, and verification commands in `checklist.md` and `context-notes.md`.

- [ ] **Step 3: Commit.**

Commit with one message describing the rubric change.
