import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.collectors.github_release_collector import GitHubReleaseCollector
from src.collectors.news_collector import NewsCollector
from src.collectors.official_ai_collector import OfficialAICollector
from src.collectors.tech_collector import TechCollector
from src.pipeline import (
    ai_impact_score,
    canonicalize_url,
    content_dedupe_key,
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
        self.assertFalse(
            is_relevant_ai_item(
                normalize_item(
                    {
                        "title": "Our response to the developer tool compromise",
                        "url": "https://example.com/incident",
                        "desc": "Security incident response and remediation details",
                        "source_type": "news",
                    }
                )
            )
        )

    def test_is_relevant_ai_item_accepts_recent_feature_signal_from_community(self):
        self.assertTrue(
            is_relevant_ai_item(
                normalize_item(
                    {
                        "title": "Has any one got FluxBridge to work?",
                        "url": "https://example.com/community-signal",
                        "desc": "People are sharing first impressions after rollout.",
                        "source_type": "community",
                        "metadata": {"recent_feature_signal": True},
                    }
                )
            )
        )

    def test_is_relevant_ai_item_accepts_core_provider_security_incident(self):
        item = normalize_item(
            {
                "title": "Anthropic investigates unauthorized access to Claude Mythos Preview",
                "url": "https://example.com/claude-mythos-breach",
                "desc": "The frontier cybersecurity model may have been accessed through a third-party vendor.",
                "source_type": "news",
            }
        )

        self.assertTrue(is_relevant_ai_item(item))
        self.assertGreater(ai_impact_score(item), 0)

    def test_deduplicate_and_rank_filters_low_signal_openai_academy_content(self):
        items = [
            {
                "title": "[Official] ChatGPT for marketing teams",
                "url": "https://openai.com/academy/marketing",
                "desc": "Learn how marketing teams use ChatGPT to plan campaigns.",
                "source_name": "OpenAI News RSS",
                "source_type": "official_news",
                "published_at": "2026-04-10T00:00:00Z",
            },
            {
                "title": "[Official] OpenAI Responses API tool update",
                "url": "https://platform.openai.com/docs/changelog#responses",
                "desc": "Responses API adds a new tool.",
                "source_name": "OpenAI API Changelog",
                "source_type": "official_news",
                "published_at": "2026-04-10T01:00:00Z",
            },
        ]

        ranked = deduplicate_and_rank(items, ["OpenAI", "Responses"], limit=10)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].url, "https://platform.openai.com/docs/changelog#responses")

    def test_deduplicate_and_rank_collapses_same_gemma_family_hf_models(self):
        items = [
            {
                "title": "[HF Model] google/gemma-4-E2B-it (100 likes)",
                "url": "https://huggingface.co/google/gemma-4-E2B-it",
                "desc": "Downloads: 1 | Task: any-to-any",
                "source_name": "Hugging Face",
                "source_type": "model_release",
                "published_at": "2026-04-10T16:35:43.000Z",
            },
            {
                "title": "[HF Model] alt/gemma-4-E2B-it-fast (50 likes)",
                "url": "https://huggingface.co/alt/gemma-4-E2B-it-fast",
                "desc": "Downloads: 2 | Task: any-to-any",
                "source_name": "Hugging Face",
                "source_type": "model_release",
                "published_at": "2026-04-10T16:35:40.000Z",
            },
        ]

        ranked = deduplicate_and_rank(items, ["Gemma"], limit=10)
        self.assertEqual(len(ranked), 1)

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

    def test_deduplicate_and_rank_prefers_core_provider_items_over_generic_items(self):
        items = [
            {
                "title": "MiniMax releases a new checkpoint",
                "url": "https://example.com/minimax",
                "source_name": "Hugging Face",
                "source_type": "model_release",
                "published_at": "2026-04-10T10:00:00Z",
            },
            {
                "title": "OpenAI ships a new Codex update",
                "url": "https://example.com/codex",
                "source_name": "OpenAI",
                "source_type": "model_release",
                "published_at": "2026-04-10T09:00:00Z",
            },
        ]

        ranked = deduplicate_and_rank(items, ["OpenAI", "Codex"], limit=10)
        self.assertEqual(ranked[0].url, "https://example.com/codex")

    def test_deduplicate_and_rank_promotes_high_impact_security_item_within_provider(self):
        items = [
            {
                "title": "Anthropic and NEC collaborate to build Japan AI workforce",
                "url": "https://example.com/anthropic-nec",
                "desc": "Enterprise partnership announcement.",
                "source_name": "Anthropic News",
                "source_type": "official_news",
                "published_at": "2026-04-24T00:00:00Z",
            },
            {
                "title": "Claude Mythos Preview found high-severity zero-day vulnerabilities",
                "url": "https://example.com/mythos",
                "desc": "Project Glasswing gives defenders access to a frontier cybersecurity model.",
                "source_name": "Anthropic Project Glasswing",
                "source_type": "official_news",
                "published_at": "2026-04-07T00:00:00Z",
            },
        ]

        ranked = deduplicate_and_rank(items, ["Claude", "Mythos", "Glasswing", "Anthropic"], limit=10)
        self.assertEqual(ranked[0].url, "https://example.com/mythos")

    def test_deduplicate_and_rank_collapses_openai_release_family_system_card(self):
        items = [
            {
                "title": "[Official] Introducing GPT-5.5",
                "url": "https://openai.com/index/introducing-gpt-5-5",
                "desc": "Introducing GPT-5.5, our smartest model yet.",
                "source_name": "OpenAI News RSS",
                "source_type": "official_news",
                "published_at": "2026-04-23T11:00:00Z",
            },
            {
                "title": "[Official] GPT-5.5 System Card",
                "url": "https://openai.com/index/gpt-5-5-system-card",
                "desc": "Safety evaluations and deployment notes for GPT-5.5.",
                "source_name": "OpenAI News RSS",
                "source_type": "official_news",
                "published_at": "2026-04-23T11:00:00Z",
            },
        ]

        ranked = deduplicate_and_rank(items, ["OpenAI", "GPT"], limit=10)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].url, "https://openai.com/index/introducing-gpt-5-5")

    def test_content_dedupe_key_uses_url_for_short_distinct_titles(self):
        capex_key = content_dedupe_key(
            title="Tesla 增加資本支出",
            url="https://techcrunch.com/tesla-capex",
            source_name="TechCrunch",
        )
        earnings_key = content_dedupe_key(
            title="Tesla財報公佈",
            url="https://techcrunch.com/tesla-earnings",
            source_name="TechCrunch",
        )

        self.assertNotEqual(capex_key, earnings_key)


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


class TestNewsCollector(unittest.TestCase):
    def test_ai_news_queries_include_high_impact_security_terms(self):
        collector = NewsCollector()
        captured_keywords: list[str] = []

        def fake_fetch(keywords, **kwargs):
            captured_keywords.extend(keywords)
            return []

        with patch.object(collector, "_fetch_by_keywords", side_effect=fake_fetch):
            collector.fetch_ai_tech_news()

        normalized = " ".join(captured_keywords).lower()
        self.assertIn("claude mythos", normalized)
        self.assertIn("project glasswing", normalized)
        self.assertIn("zero-day", normalized)
        self.assertIn("unauthorized access", normalized)

    def test_ai_news_queries_are_split_under_newsapi_q_limit(self):
        collector = NewsCollector()
        queries: list[str] = []

        def fake_fetch(keywords, **kwargs):
            queries.append(collector._build_query(keywords))
            return []

        with patch.object(collector, "_fetch_by_keywords", side_effect=fake_fetch):
            collector.fetch_ai_tech_news()

        self.assertGreater(len(queries), 2)
        self.assertTrue(all(len(query) <= collector._NEWSAPI_Q_BUDGET_CHARS for query in queries))

    @patch("src.collectors.news_collector.requests.get")
    def test_fetch_by_keywords_rejects_oversized_query_before_request(self, mock_get):
        collector = NewsCollector()
        collector.api_key = "test-key"
        long_keywords = [f"keyword-{index:03d}-with-extra-text" for index in range(30)]

        result = collector._fetch_by_keywords(
            long_keywords,
            domains="example.com",
            page_size=10,
            days_back=7,
            source_type="news",
        )

        self.assertEqual(result, [])
        mock_get.assert_not_called()


class TestOfficialCollector(unittest.TestCase):
    @patch("src.collectors.official_ai_collector.curl_requests.get")
    @patch("src.collectors.official_ai_collector.requests.get")
    def test_fetch_page_text_retries_with_curl_cffi_after_403(self, mock_get, mock_curl_get):
        mock_get.return_value = _MockResponse(text="blocked", status_code=403)
        mock_curl_get.return_value = _MockResponse(text="<html>ok</html>", status_code=200)

        collector = OfficialAICollector()
        html = collector._fetch_page_text(
            "https://platform.openai.com/docs/changelog",
            source_name="OpenAI API Changelog",
        )

        self.assertEqual(html, "<html>ok</html>")
        mock_curl_get.assert_called_once()

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

    def test_feed_updates_include_openai_rss_source(self):
        collector = OfficialAICollector()
        with patch.object(collector, "_fetch_single_feed_source", return_value=[]) as mock_feed:
            collector._fetch_feed_updates(limit_per_source=3)

        called_urls = [call.kwargs["url"] for call in mock_feed.call_args_list]
        self.assertIn("https://openai.com/news/rss.xml", called_urls)

    def test_static_official_pages_include_project_glasswing_and_red_team_mythos(self):
        collector = OfficialAICollector()

        def fake_fetch(url, *, source_name):
            if "red.anthropic.com" in url:
                return "<html>Claude Mythos vulnerability cybersecurity red team assessment</html>"
            return "<html>Project Glasswing Claude Mythos zero-day cybersecurity announcement</html>"

        with patch.object(collector, "_fetch_page_text", side_effect=fake_fetch):
            items = collector._fetch_static_official_pages()

        urls = {item["url"] for item in items}
        self.assertIn("https://www.anthropic.com/glasswing", urls)
        self.assertIn("https://red.anthropic.com/2026/mythos-preview/", urls)

    def test_extract_docs_release_notes_parses_dated_sections(self):
        html = """
        <html><body>
          <h2 id="apr-11-2026">April 11, 2026</h2>
          <ul>
            <li><a href="/docs/en/ultraplan">Ultraplan is now available in research preview</a></li>
            <li>Minor docs cleanup</li>
          </ul>
        </body></html>
        """
        collector = OfficialAICollector()
        items = collector._extract_docs_release_notes(
            source_name="Claude Code Changelog",
            source_url="https://code.claude.com/docs/en/changelog",
            html=html,
            keywords=["ultraplan", "claude code"],
            limit=5,
        )

        self.assertEqual(len(items), 1)
        self.assertIn("Ultraplan", items[0]["title"])
        self.assertEqual(items[0]["url"], "https://code.claude.com/docs/en/ultraplan")

    def test_extract_docs_release_notes_handles_version_heading_with_following_date(self):
        html = """
        <html><body>
          <h2 id="v2192">2.1.92</h2>
          <p>April 7, 2026</p>
          <ul>
            <li>/ultraplan and other remote-session features now auto-create a default cloud environment</li>
          </ul>
        </body></html>
        """
        collector = OfficialAICollector()
        items = collector._extract_docs_release_notes(
            source_name="Claude Code Changelog",
            source_url="https://code.claude.com/docs/en/changelog",
            html=html,
            keywords=["ultraplan", "claude code"],
            limit=5,
            link_hints={"ultraplan": "https://code.claude.com/docs/en/ultraplan"},
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://code.claude.com/docs/en/ultraplan")
        self.assertEqual(items[0]["published_at"], "2026-04-07T00:00:00+00:00")

    def test_extract_docs_release_notes_handles_openai_changelog_cards(self):
        html = """
        <html><body>
          <div class="mt-5">
            <div class="grid grid-cols-[3rem_1fr] items-start gap-x-4 gap-y-2">
              <div><div data-variant="outline">Jan 13</div></div>
              <div>
                <div class="flex flex-wrap gap-2 mb-2">
                  <div data-variant="soft">Feature</div>
                  <div data-variant="soft">v1/realtime</div>
                </div>
                <div class="_MarkdownContent _ChangelogMarkdown">
                  <p>Added dedicated SIP IP ranges for Realtime API.</p>
                  <a href="/api/docs/guides/realtime-sip#dedicated-sip-ip-ranges">Learn more</a>
                </div>
              </div>
            </div>
          </div>
          <div class="mt-5">
            <div class="grid grid-cols-[3rem_1fr] items-start gap-x-4 gap-y-2">
              <div><div data-variant="outline">Dec 11</div></div>
              <div>
                <div class="flex flex-wrap gap-2 mb-2">
                  <div data-variant="soft">Feature</div>
                  <div data-variant="soft">v1/responses</div>
                </div>
                <div class="_MarkdownContent _ChangelogMarkdown">
                  <p>Released GPT-5.2 to the Responses API.</p>
                  <a href="/docs/models/gpt-5.2">Read more</a>
                </div>
              </div>
            </div>
          </div>
        </body></html>
        """
        collector = OfficialAICollector()
        items = collector._extract_docs_release_notes(
            source_name="OpenAI API Changelog",
            source_url="https://platform.openai.com/docs/changelog",
            html=html,
            keywords=["responses", "realtime", "gpt"],
            limit=5,
        )

        self.assertEqual(len(items), 2)
        self.assertIn("v1/realtime", items[0]["title"])
        self.assertEqual(
            items[0]["url"],
            "https://platform.openai.com/api/docs/guides/realtime-sip#dedicated-sip-ip-ranges",
        )
        self.assertTrue(items[0]["published_at"].startswith(f"{datetime.now(timezone.utc).year}-01-13"))
        self.assertTrue(items[1]["published_at"].startswith(f"{datetime.now(timezone.utc).year - 1}-12-11"))

    @patch("src.collectors.official_ai_collector.requests.get")
    def test_html_updates_include_google_blog_gemini_app_items(self, mock_get):
        mock_get.return_value = _MockResponse(
            text="""
            <html><body>
              <div>
                <a href="/innovation-and-ai/products/gemini-app/notebooks-gemini-notebooklm/">
                  Try notebooks in Gemini to easily keep track of projects
                </a>
                <p>Apr 08, 2026</p>
              </div>
            </body></html>
            """
        )

        collector = OfficialAICollector()
        items = collector._fetch_html_updates(limit_per_source=5)

        urls = [item["url"] for item in items]
        self.assertIn(
            "https://blog.google/innovation-and-ai/products/gemini-app/notebooks-gemini-notebooklm/",
            urls,
        )

    @patch("src.collectors.official_ai_collector.requests.get")
    def test_html_updates_extract_google_blog_publish_date_from_analytics_attr(self, mock_get):
        mock_get.return_value = _MockResponse(
            text="""
            <html><body>
              <div>
                <a
                  href="/innovation-and-ai/products/gemini-app/notebooks-gemini-notebooklm/"
                  data-ga4-analytics-lead-click='{"publish_date":"2026-04-08|13:45"}'
                >
                  Try notebooks in Gemini to easily keep track of projects
                </a>
              </div>
            </body></html>
            """
        )

        collector = OfficialAICollector()
        items = collector._fetch_html_updates(limit_per_source=5)

        notebook_item = next(
            item
            for item in items
            if item["url"]
            == "https://blog.google/innovation-and-ai/products/gemini-app/notebooks-gemini-notebooklm/"
        )
        self.assertEqual(notebook_item["published_at"], "2026-04-08T00:00:00+00:00")


class TestTechCollector(unittest.TestCase):
    @patch("src.collectors.tech_collector.feedparser.parse")
    def test_fetch_reddit_keyword_matches_returns_search_hits(self, mock_parse):
        mock_parse.return_value = type(
            "Feed",
            (),
            {
                "entries": [
                    {
                        "title": "Claude Code v2.1.92 introduces Ultraplan draft",
                        "link": "https://reddit.example.com/ultraplan",
                        "summary": "Ultraplan is rolling out",
                        "published": "2026-04-09T00:00:00Z",
                    }
                ]
            },
        )()

        collector = TechCollector()
        items = collector.fetch_reddit_keyword_matches()

        self.assertTrue(any("Ultraplan" in item["title"] for item in items))
        self.assertTrue(any(item["url"] == "https://reddit.example.com/ultraplan" for item in items))
        self.assertTrue(any(item.get("metadata", {}).get("recent_feature_signal") for item in items))

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
