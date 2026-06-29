# LAN-28 Closing Inner Thought Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `next-question` and `closing-message` keep `innerThought` tied to the current user utterance, and make `closing-message` end scenario-specific situations naturally.

**Architecture:** Keep the existing response schemas unchanged. Strengthen prompts, add small repair/fallback helpers in `app/services/conversation_service.py`, and lock the behavior with service-level regression tests plus live smoke.

**Tech Stack:** Python 3.12, unittest, existing Pydantic conversation DTOs, existing deployment GitHub Actions.

## Global Constraints

- Keep `next-question` response fields unchanged.
- Keep `closing-message` response fields unchanged.
- Do not add broad new conversation heuristics unrelated to closing-message and innerThought leakage.
- Do not make `closing-message` ask a new question.
- Do not let `innerThought` preview `nextQuestion` or future topics.
- Verify with focused tests, full unittest, compileall, diff check, develop deploy, and live smoke.

---

### Task 1: Regression Tests

**Files:**
- Modify: `tests/test_conversation_service.py`

**Interfaces:**
- Consumes: `generate_next_question(request)` and `generate_closing_message(request)`.
- Produces: Tests that fail before implementation and pass after prompt/repair/fallback changes.

- [ ] **Step 1: Add next-question leakage tests**

Add tests shaped like this.

```python
def test_next_question_repairs_inner_thought_that_previews_next_fixed_question_topic(self):
    self.service.chat = lambda *args, **kwargs: json.dumps({
        "aiQuestion": "That sounds fair. Do you like parties?",
        "translatedQuestion": "그럴 수 있지. 파티 좋아해?",
        "innerThought": "역시 같이 사는 거라 정리된 방식이 좋구나. 그런데 오늘 밤 파티도 같이 갈지 궁금하네.",
        "innerThoughtType": "GOOD",
    })
    request = NextQuestionRequest.model_validate({...})
    result = self.service.generate_next_question(request)
    self.assertNotIn("파티", result.innerThought)
    self.assertNotIn("궁금", result.innerThought)
```

- [ ] **Step 2: Add closing-message context tests**

Add tests shaped like this.

```python
def test_closing_message_fallback_accepts_party_invitation_with_joining_flow(self):
    self.service.chat = lambda *args, **kwargs: "not json"
    request = ClosingMessageRequest.model_validate({... "userUtterance": "Oh, yeah. I like parties. Thank you." ...})
    result = self.service.generate_closing_message(request)
    self.assertIn("go together", result.aiMessage)
    self.assertFalse(result.aiMessage.endswith("?"))
```

- [ ] **Step 3: Verify RED**

Run focused tests.

```bash
.venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_next_question_repairs_inner_thought_that_previews_next_fixed_question_topic tests.test_conversation_service.ConversationServiceTest.test_closing_message_fallback_accepts_party_invitation_with_joining_flow
```

Expected: FAIL because the current implementation preserves future-topic innerThought when fallback repair is disabled and uses generic closing fallback.

### Task 2: Prompt And Repair Logic

**Files:**
- Modify: `app/services/conversation_service.py`

**Interfaces:**
- Consumes: `NextQuestionRequest`, `ClosingMessageRequest`, `NextQuestionResponse`, `ClosingMessageResponse`.
- Produces: Schema-compatible responses with contextual closing and current-utterance-only `innerThought`.

- [ ] **Step 1: Tighten prompts**

Update `_next_question_system_prompt()` and `_closing_message_system_prompt()` so `innerThought` cannot mention future topics or `nextQuestion`, and closing must directly answer the last AI question intent.

- [ ] **Step 2: Add closing intent helpers**

Add narrow helpers near existing fallback helpers.

```python
def _closing_intent_kind(request: ClosingMessageRequest) -> str | None:
    normalized_question = _normalize_visible_text(request.currentTurn.aiQuestion)
    normalized_utterance = _normalize_visible_text(request.currentTurn.userUtterance)
    if "party" in normalized_question:
        if _looks_like_acceptance(normalized_utterance):
            return "party_acceptance"
        if _looks_like_rejection(normalized_utterance):
            return "party_rejection"
    return None
```

- [ ] **Step 3: Add innerThought leak repair helpers**

Extend future-topic detection so `그런데`, `궁금`, and next-question Korean topic tokens cause repair even when the text does not use existing future markers.

- [ ] **Step 4: Implement minimal fallback changes**

Make `_fallback_closing_message_en()` and `_fallback_closing_message_ko()` return scenario-specific endings for party acceptance and rejection, while leaving generic fallback for unknown cases.

### Task 3: Verification, Commit, Deploy, Smoke

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Use: `scripts/lan28_edge_quality_smoke.py` or an equivalent temporary live smoke script.

**Interfaces:**
- Consumes: local tests and deployed develop server.
- Produces: commit, origin branch push, develop deploy, and smoke evidence.

- [ ] **Step 1: Run focused tests**

```bash
.venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_next_question_repairs_inner_thought_that_previews_next_fixed_question_topic tests.test_conversation_service.ConversationServiceTest.test_closing_message_fallback_accepts_party_invitation_with_joining_flow
```

- [ ] **Step 2: Run full verification**

```bash
.venv/bin/python -m unittest discover -s tests -p 'test*.py'
.venv/bin/python -m compileall -q app tests scripts
git diff --check
```

- [ ] **Step 3: Commit and deploy**

```bash
git add app/services/conversation_service.py tests/test_conversation_service.py checklist.md context-notes.md docs/superpowers/plans/2026-06-29-lan-28-closing-inner-thought-quality.md
git commit -m "fix: closing message and innerThought quality"
git push origin feat/LAN-28-closing-quality
git push origin HEAD:develop
```

- [ ] **Step 4: Live smoke**

Run at least three live API scenarios against develop and record `aiMessage`, `translatedMessage`, `innerThought`, and `innerThoughtType`.
