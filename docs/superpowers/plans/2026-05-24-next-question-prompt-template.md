# Next Question Prompt Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the next-question system prompt into a clearer template and ground few-shot menu examples in `availableOptions`.

**Architecture:** Keep the single-model next-question workflow and existing deterministic normalization. Improve only the prompt structure by separating role, schema, decision policy, slot policy, context policy, and few-shot examples.

**Tech Stack:** Python service prompt helper, Python `unittest`, Obsidian markdown experiment log.

---

### Task 1: Add Prompt Template Regression Tests

**Files:**
- Modify: `tests/test_conversation_service.py`

- [x] **Step 1: Write failing tests**

Add tests that assert the next-question system prompt has explicit sections and that menu/help few-shot examples are grounded in available options.

- [x] **Step 2: Run focused tests to verify RED**

Run: `OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest tests.test_conversation_service.ConversationServiceTest.test_next_question_prompt_uses_sectioned_template tests.test_conversation_service.ConversationServiceTest.test_next_question_prompt_grounds_menu_few_shots_in_available_options`

Expected: fail because the current prompt is one long concatenated block and still contains hardcoded menu examples not tied to available options.

### Task 2: Refactor the System Prompt

**Files:**
- Modify: `app/services/conversation_service.py`
- Modify: `tests/test_conversation_service.py`

- [x] **Step 1: Implement the sectioned prompt**

Rewrite `_next_question_system_prompt()` with labeled sections.

- [x] **Step 2: Ground few-shot examples in available options**

Change recommendation and menu examples to include available options in the input and only recommend or list those options.

- [x] **Step 3: Run focused tests to verify GREEN**

Run the focused prompt tests and existing next-question prompt tests.

### Task 3: Documentation, Verification, and Commit

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `/Users/sangmin8817/Desktop/기타 자료/Obsidian/SayNow/SayNow AI 프롬프트 실험 로그.md`

- [x] **Step 1: Record Prompt 8 in Obsidian**

Document the prompt-engineering principle, changed prompt shape, test result, and next direction.

- [x] **Step 2: Run full verification**

Run:

```bash
OPENAI_API_KEY=test-key /private/tmp/saynow-ai-venv/bin/python -m unittest discover -s tests -p 'test*.py'
/private/tmp/saynow-ai-venv/bin/python -m compileall app tests
git diff --check
```

- [x] **Step 3: Commit**

Commit message: `llm-workflow-improvement refactor: 꼬리 질문 프롬프트 템플릿 정리`
