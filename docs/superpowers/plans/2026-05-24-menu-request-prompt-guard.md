# Menu Request Prompt Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat menu-seeking utterances such as `I need a menu` as `ASSISTANCE_REQUEST` without weakening generic non-answer protection for order slots.

**Architecture:** Keep the next-question single-model workflow. Reduce the deterministic blocker so it only blocks clear non-answers and incomplete order fragments, then guide the model with concise role, schema, and few-shot examples for menu requests.

**Tech Stack:** FastAPI service helpers, Pydantic response models, Python `unittest`, Obsidian markdown experiment log.

---

### Task 1: Add Menu Request Regression Tests

**Files:**
- Modify: `tests/test_conversation_service.py`

- [x] **Step 1: Write failing tests**

Add tests proving `I need a menu`, `Can I get a menu`, and `Menu please` are assistance requests that can be grounded in `availableOptions`.

- [x] **Step 2: Run focused tests to verify RED**

Run: `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_next_question_treats_menu_need_as_assistance_request`

Expected: fail because `I need a menu` is currently blocked before LLM processing.

### Task 2: Reduce Over-Broad Guard and Update Prompt

**Files:**
- Modify: `app/services/conversation_service.py`
- Modify: `tests/test_conversation_service.py`

- [x] **Step 1: Implement minimal guard change**

Remove `menu` from the generic order object blocker and add narrow menu-request recognition to `_is_information_request`.

- [x] **Step 2: Update prompt tests**

Assert that the next-question prompt treats menu requests as assistance requests and no longer lists `menu` as a generic object blocker.

- [x] **Step 3: Update system prompt**

Keep the structured JSON schema, concise decision workflow, and few-shot style. Replace the broad `menu` blocker with menu-request examples such as `I need a menu`, `Can I get a menu`, and `Menu please`.

- [x] **Step 4: Run focused tests to verify GREEN**

Run the focused next-question tests and confirm they pass.

### Task 3: Documentation, Verification, and Commit

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `/Users/sangmin8817/Desktop/기타 자료/Obsidian/SayNow/SayNow AI 프롬프트 실험 로그.md`

- [x] **Step 1: Record the prompt-engineering decision**

Document that the change follows `prompt-engineering-patterns`: reduce over-broad negative rules, keep structured output, and use representative few-shot examples.

- [x] **Step 2: Run full verification**

Run:

```bash
OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest discover -s tests -p 'test*.py'
/private/tmp/saynow-ai-venv/bin/python -m compileall app tests
git diff --check
```

- [x] **Step 3: Commit**

Commit message: `llm-workflow-improvement fix: 메뉴 요청 guard 완화`
