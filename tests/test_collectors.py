import unittest
from datetime import datetime, timezone

from src.pipeline import (
    canonicalize_url,
    deduplicate_and_rank,
    filter_recent_items,
    is_relevant_ai_item,
    normalize_item,
    parse_published_at,
)


class TestPipeline(unittest.TestCase):
    def test_canonicalize_url_strips_tracking_params(self):
        url = "https://example.com/post/?utm_source=x&ref=y&id=1"
        self.assertEqual(canonicalize_url(url), "https://example.com/post?id=1")

    def test_deduplicate_and_rank_prefers_longer_description(self):
        items = [
            {
                "title": "OpenAI releases model",
                "url": "https://example.com/news?utm_source=a",
                "desc": "short",
                "source_name": "A",
            },
            {
                "title": "OpenAI releases model",
                "url": "https://example.com/news",
                "desc": "longer description",
                "source_name": "B",
            },
            {
                "title": "Other item",
                "url": "https://example.com/other",
                "desc": "desc",
                "source_name": "C",
            },
        ]

        ranked = deduplicate_and_rank(items, ["OpenAI"], limit=10)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].desc, "longer description")

    def test_deduplicate_and_rank_collapses_same_title_across_urls(self):
        items = [
            {
                "title": "Anthropic introduces a new Claude agent workflow for enterprises",
                "url": "https://source-a.example.com/story-1",
                "desc": "short",
            },
            {
                "title": "Anthropic introduces a new Claude agent workflow for enterprises",
                "url": "https://source-b.example.com/story-99",
                "desc": "much longer description from a mirrored source",
            },
        ]

        ranked = deduplicate_and_rank(items, ["Claude"], limit=10)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].desc, "much longer description from a mirrored source")

    def test_deduplicate_and_rank_prefers_higher_quality_source_type(self):
        items = [
            {
                "title": "OpenAI ships new release notes for agents",
                "url": "https://example.com/community-copy",
                "desc": "Long community summary",
                "source_type": "community",
                "published_at": "2026-04-12T09:00:00Z",
            },
            {
                "title": "OpenAI ships new release notes for agents",
                "url": "https://example.com/official-post",
                "desc": "Short official summary",
                "source_type": "official_news",
                "published_at": "2026-04-12T08:00:00Z",
            },
        ]

        ranked = deduplicate_and_rank(items, ["OpenAI"], limit=10)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].source_type, "official_news")

    def test_normalize_item_uses_defaults(self):
        item = normalize_item({"title": "Sample", "url": "https://example.com"})
        self.assertEqual(item.source_name, "unknown")
        self.assertEqual(item.source_type, "unknown")

    def test_is_relevant_ai_item_filters_low_value_ai_noise(self):
        self.assertFalse(
            is_relevant_ai_item(
                normalize_item(
                    {
                        "title": "Florida launches investigation into OpenAI",
                        "url": "https://example.com/noise",
                        "desc": "Probe and lawsuit coverage",
                        "source_type": "news",
                    }
                )
            )
        )

    def test_filter_recent_items_excludes_old_or_undated_items(self):
        items = [
            normalize_item(
                {
                    "title": "Fresh official post",
                    "url": "https://example.com/fresh",
                    "source_type": "official_news",
                    "published_at": "2026-04-10T08:00:00Z",
                }
            ),
            normalize_item(
                {
                    "title": "Old official post",
                    "url": "https://example.com/old",
                    "source_type": "official_news",
                    "published_at": "2026-02-20T08:00:00Z",
                }
            ),
            normalize_item(
                {
                    "title": "Undated post",
                    "url": "https://example.com/undated",
                    "source_type": "official_news",
                    "published_at": None,
                }
            ),
        ]
        filtered = filter_recent_items(
            items,
            max_age_days=14,
            now=datetime(2026, 4, 12, tzinfo=timezone.utc),
            require_published_at=True,
        )
        self.assertEqual([item.url for item in filtered], ["https://example.com/fresh"])

    def test_parse_published_at_supports_rfc2822(self):
        parsed = parse_published_at("Sun, 12 Apr 2026 02:30:00 GMT")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.date().isoformat(), "2026-04-12")
        self.assertTrue(
            is_relevant_ai_item(
                normalize_item(
                    {
                        "title": "OpenAI ships new model release notes",
                        "url": "https://example.com/release",
                        "desc": "API and reasoning improvements",
                        "source_type": "news",
                    }
                )
            )
        )
