# 한국인 영어 오류 패턴 seed catalog를 검증한다.
import unittest


class ErrorPatternCatalogTest(unittest.TestCase):

    def test_loads_article_a_pattern_as_gamifiable_low_priority_pattern(self):
        from app.services.error_pattern_catalog import get_error_pattern, prompt_error_pattern_catalog

        pattern = get_error_pattern("article_a_omission")
        prompt_catalog = prompt_error_pattern_catalog()

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.error_type, "article_a_omission")
        self.assertEqual(pattern.korean_pct, 79)
        self.assertFalse(pattern.breaks_meaning)
        self.assertEqual(pattern.correction_priority, "low")
        self.assertTrue(pattern.gamifiable)
        self.assertEqual(pattern.denominator_type, "obligatory_context")
        self.assertIn("a/an", pattern.feedback_copy)
        self.assertIn("article_a_omission", prompt_catalog)
        self.assertIn("korean_pct=79", prompt_catalog)
        self.assertIn("breaks_meaning=false", prompt_catalog)

    def test_loads_meaning_breaking_konglish_pattern_as_high_priority(self):
        from app.services.error_pattern_catalog import get_error_pattern

        pattern = get_error_pattern("konglish")

        self.assertIsNotNone(pattern)
        self.assertTrue(pattern.breaks_meaning)
        self.assertEqual(pattern.correction_priority, "high")
        self.assertFalse(pattern.gamifiable)

    def test_loads_indirect_question_word_order_as_quantitative_hook_pattern(self):
        from app.services.error_pattern_catalog import get_error_pattern, prompt_error_pattern_catalog

        pattern = get_error_pattern("indirect_question_word_order")
        prompt_catalog = prompt_error_pattern_catalog()

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.korean_pct, 40)
        self.assertTrue(pattern.gamifiable)
        self.assertIn("간접의문문", pattern.feedback_copy)
        self.assertIn("korean_pct=40", prompt_catalog)


if __name__ == "__main__":
    unittest.main()
