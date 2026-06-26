# American Korean Quality Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `AMERICAN_LEARNER` live quality smoke runner whose test cases cover common foreign Korean learner error types and whose reports expose full input and output for human review.

**Architecture:** Add one focused script under `scripts/` with structured case definitions, live HTTP execution, deterministic evaluation, JSON output, and Markdown output. Add one local unit test file that validates the case taxonomy, request contract, and raw I/O report shape without calling the network.

**Tech Stack:** Python standard library, `unittest`, existing FastAPI JSON contracts, existing `/api/v1/conversation/turn-feedback` endpoint.

## Global Constraints

- Keep `serviceAudience` set to `AMERICAN_LEARNER` in every scenario payload.
- Keep `benchmarkMessage` expected as `null` for all American learner feedback cases.
- Preserve full raw API input and output in the generated JSON and Markdown reports.
- Do not add external dependencies.
- Use RED-GREEN TDD for the script behavior.

---

### Task 1: Case Taxonomy And Report Contract

**Files:**
- Create: `tests/test_american_korean_quality_smoke.py`
- Create: `scripts/american_korean_quality_smoke.py`

**Interfaces:**
- Produces: `PATTERN_KEYS: tuple[str, ...]`
- Produces: `CASES: tuple[KoreanLearnerCase, ...]`
- Produces: `build_turn_feedback_payload(case: KoreanLearnerCase, session_id: int, turn_id: int) -> dict[str, Any]`
- Produces: `render_markdown_report(result: dict[str, Any], output_json_path: Path) -> str`

- [ ] **Step 1: Write failing tests.**

Write tests that import `scripts/american_korean_quality_smoke.py` and assert:
- all cases use one of `particle_marker`, `verb_ending_tense`, `honorific_politeness`, `word_order_modifier`, `spacing_word_boundary`.
- each pattern has at least two cases.
- generated payloads include `scenario.serviceAudience == "AMERICAN_LEARNER"`.
- every expected result has `benchmarkMessage == None`.
- Markdown reports contain `### Input`, `### Expected`, and `### Actual Output`.

- [ ] **Step 2: Run RED.**

Run:

```bash
OPENAI_API_KEY=test-key .venv/bin/python -m unittest tests.test_american_korean_quality_smoke
```

Expected: fail because the new script does not exist yet.

- [ ] **Step 3: Implement minimal script.**

Create `scripts/american_korean_quality_smoke.py` with case definitions, payload builder, evaluator, live runner, JSON writer, and Markdown writer.

- [ ] **Step 4: Run GREEN.**

Run:

```bash
OPENAI_API_KEY=test-key .venv/bin/python -m unittest tests.test_american_korean_quality_smoke
```

Expected: pass.

### Task 2: Verification

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: Task 1 script and tests.

- [ ] **Step 1: Run focused test.**

Run:

```bash
OPENAI_API_KEY=test-key .venv/bin/python -m unittest tests.test_american_korean_quality_smoke
```

Expected: pass.

- [ ] **Step 2: Run broader project checks.**

Run:

```bash
OPENAI_API_KEY=test-key .venv/bin/python -m unittest discover -s tests -p 'test*.py'
.venv/bin/python -m compileall app tests scripts
git diff --check
```

Expected: pass.

- [ ] **Step 3: Commit one logical change.**

Commit all changes with:

```bash
git add docs/superpowers/plans/2026-06-26-american-korean-quality-smoke.md checklist.md context-notes.md tests/test_american_korean_quality_smoke.py scripts/american_korean_quality_smoke.py
git commit -m "feat: 미국인 한국어 품질 스모크 추가"
```
