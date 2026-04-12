import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.collectors.github_release_collector import GitHubReleaseCollector
from src.collectors.official_ai_collector import OfficialAICollector
from src.collectors.tech_collector import TechCollector
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

    def test_canonicalize_url_preserves_fragment_anchor(self):
        url = "https://example.com/release-notes/?utm_source=x#apr-9-2026"
        self.assertEqual(
            canonicalize_url(url),
            "https://example.com/release-notes#apr-9-2026",
        )

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

    def test_deduplicate_and_rank_sorts_by_parsed_datetime_not_raw_string(self):
        items = [
            {
                "title": "Older RFC item",
                "url": "https://example.com/older-rfc",
                "source_type": "official_news",
                "published_at": "Sun, 09 Apr 2026 12:00:00 GMT",
            },
            {
                "title": "Newer ISO item",
                "url": "https://example.com/newer-iso",
                "source_type": "official_news",
                "published_at": "2026-04-10T08:00:00Z",
            },
        ]
        ranked = deduplicate_and_rank(items, ["item"], limit=10)
        self.assertEqual(ranked[0].url, "https://example.com/newer-iso")

    def test_deduplicate_and_rank_keeps_same_page_different_anchors(self):
        items = [
            {
                "title": "Claude Platform April 9, 2026: advisor tool",
                "url": "https://example.com/release-notes#apr-9-2026",
                "source_type": "official_news",
                "published_at": "2026-04-09T00:00:00Z",
            },
            {
                "title": "Claude Platform April 8, 2026: managed agents",
                "url": "https://example.com/release-notes#apr-8-2026",
                "source_type": "official_news",
                "published_at": "2026-04-08T00:00:00Z",
            },
        ]
        ranked = deduplicate_and_rank(items, ["claude"], limit=10)
        urls = {item.url for item in ranked}
        self.assertEqual(len(ranked), 2)
        self.assertIn("https://example.com/release-notes#apr-9-2026", urls)
        self.assertIn("https://example.com/release-notes#apr-8-2026", urls)


class _MockResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json_data = json_data
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestOfficialCollector(unittest.TestCase):
    @patch("src.collectors.official_ai_collector.requests.get")
    def test_claude_release_notes_uses_heading_anchor_and_link_fallback(self, mock_get):
        html = """
        <html><body>
          <h3 id="apr-9-2026">April 9, 2026</h3>
          <ul>
            <li>We've launched the advisor tool in public beta.</li>
          </ul>
          <p><a href="/docs/en/release-notes/managed-agents">Claude Managed Agents in public beta</a></p>
        </body></html>
        """
        mock_get.return_value = _MockResponse(text=html)

        collector = OfficialAICollector()
        items = collector._fetch_claude_platform_release_notes(limit=10)

        urls = [item["url"] for item in items]
        self.assertTrue(any("#apr-9-2026" in url for url in urls))
        self.assertTrue(any("/release-notes/managed-agents" in url for url in urls))

    @patch("src.collectors.official_ai_collector.requests.get")
    def test_openai_html_fetch_failure_falls_back_to_rss(self, mock_get):
        def _mock_get(url, *args, **kwargs):
            if "openai.com" in url:
                raise RuntimeError("403 Client Error")
            return _MockResponse(text="<html><body>No matching links</body></html>")

        mock_get.side_effect = _mock_get

        collector = OfficialAICollector()
        fallback_items = [
            {
                "title": "[Official] OpenAI fallback item",
                "url": "https://openai.com/news/example",
                "desc": "fallback desc",
                "source_name": "OpenAI News RSS",
                "source_type": "official_news",
                "published_at": "2026-04-12T00:00:00Z",
            }
        ]
        with patch.object(collector, "_fetch_single_feed_source", return_value=fallback_items) as mock_fallback:
            items = collector._fetch_html_updates(limit_per_source=3)

        self.assertEqual(items, fallback_items)
        mock_fallback.assert_called_once()
        self.assertEqual(
            mock_fallback.call_args.kwargs["url"],
            "https://openai.com/news/rss.xml",
        )


class TestTechCollector(unittest.TestCase):
    def test_estimate_recent_star_delta_counts_only_recent_stars(self):
        collector = TechCollector()
        since_dt = datetime(2026, 4, 8, tzinfo=timezone.utc)

        with patch.object(
            collector.session,
            "get",
            return_value=_MockResponse(
                json_data=[
                    {"starred_at": "2026-04-10T00:00:00Z"},
                    {"starred_at": "2026-04-09T10:00:00Z"},
                    {"starred_at": "2026-04-01T10:00:00Z"},
                ]
            ),
        ):
            delta = collector._estimate_recent_star_delta("owner/repo", since_dt, max_pages=2)

        self.assertEqual(delta, 2)


class TestGitHubReleaseCollector(unittest.TestCase):
    def test_normalize_release_excerpt_strips_markdown_noise(self):
        collector = GitHubReleaseCollector()
        excerpt = collector._normalize_release_excerpt(
            "```python\nx=1\n```\n- Small patch to fix `device_map` parsing."
        )
        self.assertNotIn("```", excerpt)
        self.assertNotIn("`", excerpt)
        self.assertIn("device_map", excerpt)

    def test_low_signal_patch_release_is_filtered(self):
        collector = GitHubReleaseCollector()
        release = {
            "name": "Patch release: v5.5.3",
            "tag_name": "v5.5.3",
            "body": "Small patch release to fix parser bug.",
        }
        self.assertTrue(collector._is_low_signal_patch_release(release))

    def test_high_signal_release_is_not_filtered(self):
        collector = GitHubReleaseCollector()
        release = {
            "name": "Release v2.0.0",
            "tag_name": "v2.0.0",
            "body": "Adds new managed agent API and model routing support.",
        }
        self.assertFalse(collector._is_low_signal_patch_release(release))
