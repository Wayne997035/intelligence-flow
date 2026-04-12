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
