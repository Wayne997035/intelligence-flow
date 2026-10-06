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


class TestGptReleaseFamilies(unittest.TestCase):
    def test_gpt6_variants_are_distinct_families(self):
        from src.pipeline import content_dedupe_key, openai_release_family

        self.assertEqual(openai_release_family("introducing gpt-6 astra"), "openai-gpt-6-astra")
        self.assertEqual(openai_release_family("gpt-6 sol and luna"), "openai-gpt-6-sol")
        self.assertEqual(openai_release_family("gpt-6 is here"), "openai-gpt-6")
        astra = content_dedupe_key(title="Introducing GPT-6 Astra", url="https://openai.com/index/gpt-6-astra")
        sol = content_dedupe_key(title="Introducing GPT-6 Sol", url="https://openai.com/index/gpt-6-sol")
        self.assertNotEqual(astra, sol)

    def test_deduplicate_keeps_astra_and_sol_separately(self):
        from src.pipeline import deduplicate_and_rank

        ranked = deduplicate_and_rank(
            [
                {"title": "Introducing GPT-6 Astra", "url": "https://openai.com/index/gpt-6-astra", "source_name": "OpenAI", "source_type": "official_news", "published_at": "2026-09-04T00:00:00Z"},
                {"title": "Introducing GPT-6 Sol", "url": "https://openai.com/index/gpt-6-sol", "source_name": "OpenAI", "source_type": "official_news", "published_at": "2026-09-22T00:00:00Z"},
            ],
            ["GPT"],
            limit=10,
        )
        self.assertEqual(len(ranked), 2)

    def test_gpt6_launch_is_relevant_and_high_impact(self):
        item = _item("OpenAI rolls out GPT-6 Astra", "Available to developers via software APIs.", "news")
        self.assertTrue(is_relevant_ai_item(item))
        self.assertGreater(ai_impact_score(item), 0)


class TestLiveSmokeRegressions(unittest.TestCase):
    """Cases observed in the live smoke run on 2026-10-06."""

    def test_provider_published_post_without_keywords_is_relevant(self):
        item = normalize_item(
            {"title": "Introducing dots", "url": "https://openai.com/index/dots", "source_name": "OpenAI", "source_type": "news"}
        )
        self.assertTrue(is_relevant_ai_item(item))

    def test_fable_and_mythos_launches_are_detected(self):
        from src.pipeline import is_model_launch

        self.assertTrue(is_model_launch("Introducing Claude Fable 5.1 and Claude Mythos 5.1"))
        self.assertTrue(is_model_launch("Google rolls out Gemini 4 Argon"))

    def test_model_launch_ranks_ahead_of_subreddit_named_chatter(self):
        from src.pipeline import deduplicate_and_rank

        ranked = deduplicate_and_rank(
            [
                {"title": "[Reddit r/ClaudeAI] Update: my human has been nerfed again", "url": "https://reddit.test/1", "source_type": "community", "published_at": "2026-10-05T00:00:00Z"},
                {"title": "Google rolls out Gemini 4 Argon, its most advanced AI model", "url": "https://news.test/2", "source_type": "news", "published_at": "2026-09-30T00:00:00Z"},
            ],
            ["Claude", "Gemini"],
            limit=10,
        )
        self.assertIn("Gemini 4 Argon", ranked[0].title)


class TestArticleDateFallback(unittest.TestCase):
    def test_extracts_published_time_from_article_meta(self):
        from src.collectors.official_ai_collector import OfficialAICollector

        collector = OfficialAICollector()
        html = '<html><head><meta property="article:published_time" content="2026-10-01T16:00:00+00:00"></head></html>'
        self.assertTrue(collector._extract_article_published_at(html).startswith("2026-10-01"))
        json_ld = '<script type="application/ld+json">{"datePublished": "2026-09-30T08:00:00Z"}</script>'
        self.assertTrue(collector._extract_article_published_at(json_ld).startswith("2026-09-30"))
        self.assertIsNone(collector._extract_article_published_at("<html></html>"))

    def test_undated_feed_entry_uses_article_date(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.collectors import official_ai_collector
        from src.collectors.official_ai_collector import OfficialAICollector

        collector = OfficialAICollector()
        feed = SimpleNamespace(entries=[{"title": "Introducing Gemini agents", "link": "https://developers.googleblog.com/x", "summary": "gemini agent"}])
        with patch.object(official_ai_collector, "feedparser") as mock_feedparser, patch.object(
            collector, "_fetch_article_published_at", return_value="2026-10-01T00:00:00+00:00"
        ) as mock_lookup:
            mock_feedparser.parse.return_value = feed
            items = collector._fetch_single_feed_source(
                source_name="Google Developers Blog", url="https://x.test/feed", keywords=["gemini"], limit=4
            )
        mock_lookup.assert_called_once()
        self.assertEqual(items[0]["published_at"], "2026-10-01T00:00:00+00:00")

    def test_gone_microsoft_feed_removed(self):
        from src.collectors.official_ai_collector import OfficialAICollector

        names = [source["name"] for source in OfficialAICollector().feed_sources]
        self.assertNotIn("Microsoft AI Blog", names)
