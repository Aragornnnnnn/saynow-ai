# 미국인 한국어 학습자 품질 스모크 케이스와 리포트 계약을 검증한다.
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "american_korean_quality_smoke.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("american_korean_quality_smoke", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("american_korean_quality_smoke module spec could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AmericanKoreanQualitySmokeTest(unittest.TestCase):

    def setUp(self):
        self.smoke = _load_smoke_module()

    def test_cases_cover_standardized_foreign_korean_error_patterns(self):
        pattern_keys = set(self.smoke.PATTERN_KEYS)
        case_counts = Counter(case.pattern_key for case in self.smoke.CASES)

        self.assertEqual(
            pattern_keys,
            {
                "particle_marker",
                "verb_ending_tense",
                "honorific_politeness",
                "word_order_modifier",
                "spacing_word_boundary",
            },
        )
        self.assertEqual(set(case_counts), pattern_keys)
        for pattern_key in pattern_keys:
            self.assertGreaterEqual(case_counts[pattern_key], 2)

    def test_turn_feedback_payload_marks_every_case_as_american_learner(self):
        for index, case in enumerate(self.smoke.CASES, start=1):
            with self.subTest(case_id=case.case_id):
                payload = self.smoke.build_turn_feedback_payload(
                    case,
                    session_id=170000 + index,
                    turn_id=170000000 + index,
                )

                self.assertEqual(payload["scenario"]["serviceAudience"], "AMERICAN_LEARNER")
                self.assertEqual(payload["turn"]["aiQuestion"], case.ai_question)
                self.assertEqual(payload["turn"]["translatedQuestion"], case.translated_question)
                self.assertEqual(payload["turn"]["userUtterance"], case.user_utterance)

    def test_session_feedback_payload_uses_same_american_learner_scenario_and_expected_turn(self):
        case = self.smoke.CASES[0]

        payload = self.smoke.build_session_feedback_payload(
            case,
            session_id=170001,
            expected_turn_ids=[170001001],
        )

        self.assertEqual(payload["sessionId"], 170001)
        self.assertEqual(payload["expectedTurnIds"], [170001001])
        self.assertEqual(payload["scenario"]["serviceAudience"], "AMERICAN_LEARNER")
        self.assertEqual(payload["scenario"]["counterpartRole"], case.counterpart_role)

    def test_expected_outputs_keep_benchmark_message_null(self):
        for case in self.smoke.CASES:
            with self.subTest(case_id=case.case_id):
                self.assertEqual(case.expected_feedback_type, "NEEDS_IMPROVEMENT")
                self.assertIsNone(case.expected_benchmark_message)
                self.assertTrue(case.expected_correction_contains)

    def test_markdown_report_preserves_case_input_expected_and_actual_output(self):
        case = self.smoke.CASES[0]
        payload = self.smoke.build_turn_feedback_payload(case, session_id=170001, turn_id=170001001)
        result = {
            "metadata": {
                "executedAt": "2026-06-26T00:00:00+00:00",
                "baseUrl": "http://example.test",
                "caseCount": 1,
                "fatalIssueCount": 0,
                "reviewNoteCount": 0,
            },
            "cases": [
                {
                    "caseId": case.case_id,
                    "patternKey": case.pattern_key,
                    "purpose": case.purpose,
                    "input": payload,
                    "expected": self.smoke.expected_output(case),
                    "actualOutput": {
                        "feedbackType": "NEEDS_IMPROVEMENT",
                        "benchmarkMessage": None,
                        "correctionExpression": "저는 학교에 가요.",
                        "correctionReason": "The particle 에 marks the destination.",
                    },
                    "fatalIssues": [],
                    "reviewNotes": [],
                }
            ],
            "fatalIssues": [],
            "reviewNotes": [],
        }

        markdown = self.smoke.render_markdown_report(result, Path("/private/tmp/result.json"))

        self.assertIn("### Input", markdown)
        self.assertIn("### Expected", markdown)
        self.assertIn("### Actual Output", markdown)
        self.assertIn('"serviceAudience": "AMERICAN_LEARNER"', markdown)
        self.assertIn('"benchmarkMessage": null', markdown)


if __name__ == "__main__":
    unittest.main()
