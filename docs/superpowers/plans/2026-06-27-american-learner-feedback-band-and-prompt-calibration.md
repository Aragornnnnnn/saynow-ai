# American Learner Feedback Band And Prompt Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AMERICAN_LEARNER feedback quality and session score semantics should match the product contract without hardcoding single user utterances.

**Architecture:** Keep the existing structured JSON output and server validation. Add score band clamping based on GOOD count in the session, and improve AMERICAN_LEARNER turn-feedback prompt policy with contrastive few-shot examples for relationship fit and minimal particle correction.

**Tech Stack:** Python, Pydantic models, `unittest`, existing `app/services/conversation_service.py` prompt and scoring helpers.

## Global Constraints

- Do not add utterance-specific hardcoded classifiers for the fan-sign and blind-date prompt-quality issues.
- Preserve `benchmarkMessage=null` for `AMERICAN_LEARNER`.
- Use TDD. Write failing tests before implementation.
- Keep changes surgical in `app/services/conversation_service.py`, `tests/test_conversation_service.py`, `checklist.md`, and `context-notes.md`.

---

### Task 1: Session Score Band Contract

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/app/services/conversation_service.py`
- Modify: `/Users/sangmin8817/Soma/saynow-ai/tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `_postprocess_native_score_for_feedback_mix(native_score, turn_feedback_entries, service_audience)`.
- Produces: `nativeScore` clamped by GOOD count.

- [ ] **Step 1: Write failing tests**

Add tests that cache four turn feedbacks and assert bands.

```python
def test_session_feedback_clamps_four_turn_score_by_good_count_bands(self):
    cases = [
        (["NEEDS_IMPROVEMENT"] * 4, 99, 50),
        (["GOOD", "NEEDS_IMPROVEMENT", "NEEDS_IMPROVEMENT", "NEEDS_IMPROVEMENT"], 99, 64),
        (["GOOD", "GOOD", "NEEDS_IMPROVEMENT", "NEEDS_IMPROVEMENT"], 99, 74),
        (["GOOD", "GOOD", "GOOD", "NEEDS_IMPROVEMENT"], 99, 89),
        (["GOOD", "GOOD", "GOOD", "GOOD"], 40, 90),
    ]
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_session_feedback_clamps_four_turn_score_by_good_count_bands`.

Expected: FAIL because only all-NEEDS has a 68 cap today.

- [ ] **Step 3: Implement band clamp**

Add a helper that clamps by GOOD count.

```python
score_bands = {
    0: (50, 50),
    1: (55, 64),
    2: (65, 74),
    3: (75, 89),
    4: (90, 100),
}
```

- [ ] **Step 4: Run GREEN**

Run the focused band tests and existing session feedback tests.

### Task 2: AMERICAN_LEARNER Prompt Calibration

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/app/services/conversation_service.py`
- Modify: `/Users/sangmin8817/Soma/saynow-ai/tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `_american_learner_turn_feedback_system_prompt()`.
- Produces: prompt policy and examples for fan-sign warmth, blind-date ride refusal, and particle minimal correction.

- [ ] **Step 1: Write failing prompt-policy tests**

Assert that the prompt contains all new policy markers and examples.

```python
self.assertIn("Relationship Fit Gate", system_prompt)
self.assertIn("Minimal Particle Correction Gate", system_prompt)
self.assertIn("당연하지. 뭐하고 지냈어? 너무 보고 싶었어.", system_prompt)
self.assertIn("뭐하고 지냈어?", system_prompt)
self.assertIn("엥? 우리 처음 봤는데?", system_prompt)
self.assertIn("아, 감사하지만 오늘은 제가 따로 갈게요 ㅎㅎ.", system_prompt)
self.assertIn("나는 콘서트를 꼭 가보고 싶어", system_prompt)
self.assertIn("do not remove a valid object particle", system_prompt)
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_american_learner_turn_feedback_prompt_includes_scenario_pragmatics_rubric`.

Expected: FAIL because the new prompt markers and examples are not present yet.

- [ ] **Step 3: Implement prompt-only calibration**

Add concise policy and contrastive examples. Do not add `if "엥"` or `if "당연하지"` classifiers.

- [ ] **Step 4: Run GREEN**

Run the focused prompt tests and nearby AMERICAN_LEARNER tests.

### Task 3: Verification And Commit

**Files:**
- Modify: `/Users/sangmin8817/Soma/saynow-ai/checklist.md`
- Modify: `/Users/sangmin8817/Soma/saynow-ai/context-notes.md`

**Interfaces:**
- Consumes: local tests and compile checks.
- Produces: verified commit.

- [ ] **Step 1: Run focused tests**

Run the score band test and AMERICAN_LEARNER prompt tests.

- [ ] **Step 2: Run full checks**

Run `.venv/bin/python -m unittest tests.test_conversation_service`, `.venv/bin/python -m unittest discover -s tests -p 'test*.py'`, `.venv/bin/python -m compileall app tests scripts`, and `git diff --check`.

- [ ] **Step 3: Commit**

Commit one logical change after verification.
