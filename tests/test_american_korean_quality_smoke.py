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

    def test_scenario_quality_cases_preserve_user_provided_scenarios(self):
        cases = getattr(self.smoke, "SCENARIO_CASES", ())

        self.assertEqual([case.case_id for case in cases], [
            "SQ-FANSIGN",
            "SQ-FAN-FRIEND",
            "SQ-DATE",
        ])
        self.assertEqual([len(case.turns) for case in cases], [4, 4, 5])
        self.assertIn("어떻게 이렇게 잘해", cases[0].turns[1].ai_question)
        self.assertEqual(cases[0].turns[1].user_utterance, "네, 저 한국어 잘해요.")
        self.assertIn("콘서트 같이 갈래", cases[1].turns[3].ai_question)
        self.assertEqual(cases[1].turns[3].user_utterance, "네, 같이 가고 싶습니다.")
        self.assertEqual(cases[2].turns[3].turn_key, "Q4_ACCEPT")
        self.assertEqual(cases[2].turns[4].turn_key, "Q4_REJECT")

    def test_scenario_quality_payloads_use_american_learner_and_fixed_question_flow(self):
        case = self.smoke.SCENARIO_CASES[0]
        turn_feedback_payload = self.smoke.build_scenario_turn_feedback_payload(
            case,
            case.turns[0],
            session_id=190001,
            turn_id=190001001,
        )
        next_question_payload = self.smoke.build_scenario_next_question_payload(
            case,
            turn_index=0,
            session_id=190001,
            turn_id=190001001,
        )
        closing_payload = self.smoke.build_scenario_closing_message_payload(
            case,
            case.turns[-1],
            session_id=190001,
            turn_id=190001004,
        )

        self.assertEqual(turn_feedback_payload["scenario"]["serviceAudience"], "AMERICAN_LEARNER")
        self.assertEqual(next_question_payload["scenario"]["serviceAudience"], "AMERICAN_LEARNER")
        self.assertEqual(closing_payload["scenario"]["serviceAudience"], "AMERICAN_LEARNER")
        self.assertEqual(next_question_payload["currentTurn"]["aiQuestion"], case.turns[0].ai_question)
        self.assertEqual(next_question_payload["nextQuestion"]["questionKo"], case.turns[1].ai_question)
        self.assertEqual(closing_payload["currentTurn"]["userUtterance"], case.turns[-1].user_utterance)

    def test_scenario_quality_expected_outputs_keep_benchmark_null_and_corrections(self):
        for case in self.smoke.SCENARIO_CASES:
            for turn in case.turns:
                with self.subTest(case_id=case.case_id, turn_key=turn.turn_key):
                    expected = self.smoke.expected_scenario_turn_output(turn)

                    self.assertIsNone(expected["benchmarkMessage"])
                    if turn.expected_feedback_type == "NEEDS_IMPROVEMENT":
                        self.assertTrue(expected["correctionExpressionContainsAny"])

    def test_scenario_markdown_report_preserves_turn_input_expected_and_actual_output(self):
        case = self.smoke.SCENARIO_CASES[0]
        turn = case.turns[1]
        turn_feedback_payload = self.smoke.build_scenario_turn_feedback_payload(
            case,
            turn,
            session_id=190001,
            turn_id=190001002,
        )
        result = {
            "metadata": {
                "executedAt": "2026-06-26T00:00:00+00:00",
                "baseUrl": "http://example.test",
                "scenarioCount": 1,
                "turnCaseCount": 1,
                "fatalIssueCount": 0,
                "reviewNoteCount": 0,
                "dryRun": False,
            },
            "scenarios": [
                {
                    "caseId": case.case_id,
                    "title": case.title,
                    "scenario": self.smoke._scenario_quality_payload(case),
                    "turns": [
                        {
                            "turnKey": turn.turn_key,
                            "purpose": turn.purpose,
                            "input": {
                                "turnFeedbackRequest": turn_feedback_payload,
                            },
                            "expected": self.smoke.expected_scenario_turn_output(turn),
                            "actualOutput": {
                                "turnFeedback": {
                                    "feedbackType": "NEEDS_IMPROVEMENT",
                                    "benchmarkMessage": None,
                                    "correctionExpression": "아직 부족하지만 열심히 공부하고 있어요.",
                                    "correctionReason": "This sounds more modest and natural in this fan-sign context.",
                                }
                            },
                            "fatalIssues": [],
                            "reviewNotes": [],
                        }
                    ],
                    "fatalIssues": [],
                    "reviewNotes": [],
                }
            ],
            "fatalIssues": [],
            "reviewNotes": [],
        }

        markdown = self.smoke.render_scenario_markdown_report(result, Path("/private/tmp/result.json"))

        self.assertIn("## Scenario 1. SQ-FANSIGN", markdown)
        self.assertIn("### Turn Q2", markdown)
        self.assertIn("### Input", markdown)
        self.assertIn("### Expected", markdown)
        self.assertIn("### Actual Output", markdown)
        self.assertIn('"serviceAudience": "AMERICAN_LEARNER"', markdown)
        self.assertIn('"benchmarkMessage": null', markdown)


if __name__ == "__main__":
    unittest.main()
