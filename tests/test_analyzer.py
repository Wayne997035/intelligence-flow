import unittest

from src.ai.analyzer import AIAnalyzer
from src.models import IntelligenceItem


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
