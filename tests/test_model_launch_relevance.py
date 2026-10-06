import unittest

from src.ai.analyzer import AIAnalyzer
from src.pipeline import ai_impact_score, contains_term, is_relevant_ai_item, normalize_item


def _item(title, desc="", source_type="official_news"):
    return normalize_item({"title": title, "url": f"https://x.test/{title}", "desc": desc, "source_type": source_type})


class TestModelLaunchRelevance(unittest.TestCase):
    def test_official_launch_mentioning_software_is_kept(self):
        # Regression: negative keyword "war" used to match "software"/"forward".
        for title, desc in (
            ("[Official] Introducing Claude Opus 5.5", "Best model for software engineering."),
            ("[Official] Introducing GPT-5.5", "A step forward in reasoning."),
            ("Gemini 3.5 Pro now available", "Hardware-aware inference, award winning team."),
        ):
            with self.subTest(title=title):
                self.assertTrue(is_relevant_ai_item(_item(title, desc)))

    def test_model_launch_gets_high_impact_score(self):
        self.assertGreater(ai_impact_score(_item("Anthropic releases Claude Opus 5.5", source_type="news")), 0)
        self.assertEqual(ai_impact_score(_item("Weekly AI roundup", source_type="news")), 0)

    def test_negative_keywords_still_filter_whole_words(self):
        self.assertFalse(is_relevant_ai_item(_item("Department of War signs AI contract", source_type="news")))
        self.assertFalse(is_relevant_ai_item(_item("Man arrested after AI scam", source_type="news")))

    def test_contains_term_requires_word_boundary(self):
        self.assertFalse(contains_term("software engineering", "war"))
        self.assertTrue(contains_term("trade war escalates", "war"))
        self.assertFalse(contains_term("ai hackathon winners", "hack"))


class TestResponseParsing(unittest.TestCase):
    def test_parse_response_tolerates_prose_around_json(self):
        raw = 'Here is the report:\n{"summary": "s", "items": [{"title": "GPT-5.5", "url": "https://openai.com/index/introducing-gpt-5-5/", "summary": "x", "insight": "y"}], "outlook": "o"}\nThanks!'
        report = AIAnalyzer(enable_ai=False)._parse_response(raw, title="t", outlook_label="l")
        self.assertIsNotNone(report)
        self.assertEqual(report.summary, "s")


if __name__ == "__main__":
    unittest.main()
