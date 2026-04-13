import unittest

from src.ai.analyzer import AIAnalyzer
from src.models import AnalyzedReport, IntelligenceItem, ReportItem


class TestAnalyzer(unittest.TestCase):
    def test_fallback_ai_report_is_structured(self):
        analyzer = AIAnalyzer(enable_ai=False)
        report = analyzer.analyze_ai_tech(
            [
                IntelligenceItem(
                    title="OpenAI release",
                    url="https://example.com/openai",
                    desc="New model details",
                    source_name="OpenAI",
                    source_type="official_news",
                )
            ]
        )

        self.assertEqual(report.title, "AI 技術前沿情報")
        self.assertEqual(len(report.items), 1)
        self.assertIn("OpenAI", report.summary)

    def test_fallback_stock_report_contains_quotes(self):
        analyzer = AIAnalyzer(enable_ai=False)
        report = analyzer.analyze_stock_market(
            [{"symbol": "NVDA", "price": 900.0, "change": "+10.5", "range": "880-905"}],
            [
                IntelligenceItem(
                    title="NVIDIA earnings preview",
                    url="https://example.com/nvda",
                    desc="Demand remains strong",
                    source_name="Reuters",
                    source_type="news",
                )
            ],
        )

        self.assertIn("NVDA", report.summary)
        self.assertEqual(report.outlook_label, "🕵️ 專家總結")

    def test_post_process_ai_report_keeps_high_signal_source_coverage(self):
        analyzer = AIAnalyzer(enable_ai=False)
        report = AnalyzedReport(
            title="AI 技術前沿情報",
            summary="x",
            items=[
                ReportItem(
                    title="Some community summary",
                    url="https://example.com/community",
                    summary="s",
                    insight="i",
                    source_name="unknown",
                    source_type="unknown",
                    published_at=None,
                )
            ],
            outlook="o",
            outlook_label="🔮 未來展望",
        )
        news = [
            IntelligenceItem(
                title="Community item",
                url="https://example.com/community",
                desc="desc",
                source_name="Hacker News",
                source_type="community",
                published_at="2026-04-12T08:00:00Z",
            ),
            IntelligenceItem(
                title="Official launch",
                url="https://example.com/official",
                desc="desc",
                source_name="Anthropic",
                source_type="official_news",
                published_at="2026-04-12T09:00:00Z",
            ),
            IntelligenceItem(
                title="Model release",
                url="https://example.com/model",
                desc="desc",
                source_name="Hugging Face",
                source_type="model_release",
                published_at="2026-04-12T10:00:00Z",
            ),
            IntelligenceItem(
                title="SDK release",
                url="https://example.com/release",
                desc="desc",
                source_name="GitHub Releases",
                source_type="github_release",
                published_at="2026-04-12T11:00:00Z",
            ),
        ]

        processed = analyzer._post_process_ai_report(report, news)
        source_types = {item.source_type for item in processed.items}

        self.assertIn("official_news", source_types)
        self.assertIn("model_release", source_types)
        self.assertIn("github_release", source_types)
        self.assertIn("community", source_types)
        self.assertEqual(processed.items[0].source_type, "official_news")

    def test_parse_response_handles_non_string_fields(self):
        analyzer = AIAnalyzer(enable_ai=False)
        raw = """{
          "summary": {"text": "summary from object"},
          "items": [
            {
              "title": {"text": "title from object"},
              "url": "https://example.com/item",
              "summary": {"content": "item summary"},
              "insight": ["insight", "from", "list"],
              "source_name": {"value": "ExampleSource"},
              "source_type": "official_news",
              "published_at": {"text": "2026-04-12T00:00:00Z"}
            }
          ],
          "outlook": {"content": "outlook from object"}
        }"""
        parsed = analyzer._parse_response(raw, title="AI 技術前沿情報", outlook_label="🔮 未來展望")

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.summary, "summary from object")
        self.assertEqual(parsed.items[0].title, "title from object")
        self.assertEqual(parsed.items[0].summary, "item summary")
        self.assertEqual(parsed.items[0].insight, "insight from list")
        self.assertEqual(parsed.items[0].source_name, "ExampleSource")
        self.assertEqual(parsed.items[0].published_at, "2026-04-12T00:00:00Z")

    def test_post_process_ai_report_prefers_distinct_official_sources(self):
        analyzer = AIAnalyzer(enable_ai=False)
        report = AnalyzedReport(
            title="AI 技術前沿情報",
            summary="x",
            items=[
                ReportItem(
                    title="Community only",
                    url="https://example.com/community-only",
                    summary="s",
                    insight="i",
                    source_name="Reddit",
                    source_type="community",
                    published_at="2026-04-12T08:00:00Z",
                )
            ],
            outlook="o",
            outlook_label="🔮 未來展望",
        )
        news = [
            IntelligenceItem(
                title="[Official] Latest official update A",
                url="https://example.com/official/latest#a",
                desc="official update a",
                source_name="Claude Platform Release Notes",
                source_type="official_news",
                published_at="2026-04-09T00:00:00Z",
            ),
            IntelligenceItem(
                title="[Official] Latest official update B",
                url="https://example.com/official/latest#b",
                desc="official update b",
                source_name="Claude Platform Release Notes",
                source_type="official_news",
                published_at="2026-04-08T00:00:00Z",
            ),
            IntelligenceItem(
                title="[Official] Another source update",
                url="https://example.com/another-official",
                desc="other official",
                source_name="Other Official Source",
                source_type="official_news",
                published_at="2026-04-07T00:00:00Z",
            ),
        ]

        processed = analyzer._post_process_ai_report(report, news)
        urls = {item.url for item in processed.items}

        self.assertIn("https://example.com/official/latest#a", urls)
        self.assertIn("https://example.com/another-official", urls)
        self.assertNotIn("https://example.com/official/latest#b", urls)

    def test_post_process_ai_report_backfills_core_provider_coverage(self):
        analyzer = AIAnalyzer(enable_ai=False)
        report = AnalyzedReport(
            title="AI 技術前沿情報",
            summary="x",
            items=[
                ReportItem(
                    title="Anthropic item only",
                    url="https://example.com/anthropic",
                    summary="s",
                    insight="i",
                    source_name="Anthropic",
                    source_type="official_news",
                    published_at="2026-04-12T08:00:00Z",
                )
            ],
            outlook="o",
            outlook_label="🔮 未來展望",
        )
        news = [
            IntelligenceItem(
                title="Anthropic item only",
                url="https://example.com/anthropic",
                desc="desc",
                source_name="Anthropic",
                source_type="official_news",
                published_at="2026-04-12T08:00:00Z",
            ),
            IntelligenceItem(
                title="OpenAI Responses update",
                url="https://platform.openai.com/docs/changelog#responses",
                desc="desc",
                source_name="OpenAI API Changelog",
                source_type="official_news",
                published_at="2026-04-12T09:00:00Z",
            ),
            IntelligenceItem(
                title="Gemini API notes",
                url="https://ai.google.dev/gemini-api/docs/changelog#gemini",
                desc="desc",
                source_name="Gemini API Release Notes",
                source_type="official_news",
                published_at="2026-04-12T10:00:00Z",
            ),
            IntelligenceItem(
                title="Grok API notes",
                url="https://docs.x.ai/developers/release-notes#grok",
                desc="desc",
                source_name="xAI Release Notes",
                source_type="official_news",
                published_at="2026-04-12T11:00:00Z",
            ),
        ]

        processed = analyzer._post_process_ai_report(report, news)
        provider_sources = {item.source_name for item in processed.items}

        self.assertIn("Anthropic", provider_sources)
        self.assertIn("OpenAI API Changelog", provider_sources)
        self.assertIn("Gemini API Release Notes", provider_sources)
        self.assertIn("xAI Release Notes", provider_sources)

    def test_build_ai_brief_item_generates_chinese_summary_for_model_release(self):
        analyzer = AIAnalyzer(enable_ai=False)
        brief = analyzer.build_ai_brief_item(
            IntelligenceItem(
                title="[HF Model] google/gemma-4-E2B-it (417 likes)",
                url="https://huggingface.co/google/gemma-4-E2B-it",
                desc="Downloads: 857206 | Task: any-to-any",
                source_name="Hugging Face",
                source_type="model_release",
                published_at="2026-04-10T16:35:43.000Z",
            )
        )

        self.assertIn("模型更新重點", brief["summary"])
        self.assertIn("任務類型", brief["summary"])

    def test_build_ai_brief_item_generates_chinese_summary_for_community(self):
        analyzer = AIAnalyzer(enable_ai=False)
        brief = analyzer.build_ai_brief_item(
            IntelligenceItem(
                title="[Reddit r/ClaudeAI] Has any one got UltraPlan to work?",
                url="https://reddit.example.com/ultraplan",
                desc="People are sharing first impressions after rollout.",
                source_name="Reddit/ClaudeAI",
                source_type="community",
                published_at="2026-04-08T14:16:30+00:00",
            )
        )

        self.assertIn("社群近期討論焦點", brief["summary"])
        self.assertIn("Anthropic", brief["insight"])

    def test_build_ai_brief_item_generates_specific_insight_for_advisor_tool(self):
        analyzer = AIAnalyzer(enable_ai=False)
        brief = analyzer.build_ai_brief_item(
            IntelligenceItem(
                title="[Official] Claude Platform April 9, 2026: We've launched the advisor tool in public beta.",
                url="https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool",
                desc="Advisor pairs a strategic model with a faster executor model.",
                source_name="Claude Platform Release Notes",
                source_type="official_news",
                published_at="2026-04-09T00:00:00Z",
            )
        )

        self.assertIn("雙模型協作", brief["insight"])
        self.assertIn("Agent workflow", brief["insight"])

    def test_build_ai_brief_item_generates_specific_insight_for_chatgpt_pro(self):
        analyzer = AIAnalyzer(enable_ai=False)
        brief = analyzer.build_ai_brief_item(
            IntelligenceItem(
                title="ChatGPT has a new $100 per month Pro subscription",
                url="https://www.theverge.com/ai-artificial-intelligence/909599/chatgpt-pro-subscription-new",
                desc="OpenAI announced a new Pro tier with more Codex usage.",
                source_name="The Verge",
                source_type="news",
                published_at="2026-04-09T22:57:15Z",
            )
        )

        self.assertIn("OpenAI", brief["insight"])
        self.assertIn("商業化", brief["insight"])
