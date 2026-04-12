import unittest
from datetime import datetime, timezone

from src.deliverers.discord_sender import DiscordSender
from src.deliverers.notion_sender import NotionSender
from src.models import AnalyzedReport, ReportItem


def build_report() -> AnalyzedReport:
    return AnalyzedReport(
        title="AI 技術前沿情報",
        summary="這是摘要",
        items=[
            ReportItem(
                title="測試標題1",
                url="https://example.com/news1",
                summary="這是測試摘要1",
                insight="這是測試洞察1",
                source_name="Example",
                source_type="official_news",
                published_at="2026-04-12T10:00:00Z",
            ),
            ReportItem(
                title="測試標題2",
                url="https://example.com/news2",
                summary="這是測試摘要2",
                insight="這是測試洞察2",
                source_name="GitHub",
                source_type="github_release",
            ),
            ReportItem(
                title="測試標題3",
                url="https://example.com/news3",
                summary="這是測試摘要3",
                insight="這是測試洞察3",
                source_name="Anthropic",
                source_type="official_news",
            ),
            ReportItem(
                title="測試標題4",
                url="https://example.com/news4",
                summary="這是測試摘要4",
                insight="這是測試洞察4",
                source_name="arXiv",
                source_type="research",
            ),
        ],
        outlook="這是未來展望",
        outlook_label="🔮 未來展望",
    )


class TestDeliverers(unittest.TestCase):
    def test_notion_title_uses_taiwan_timezone(self):
        sender = NotionSender(dry_run=True)
        title = sender._build_title(
            "[AI 技術]",
            now=datetime(2026, 4, 12, 16, 5, tzinfo=timezone.utc),
        )
        self.assertEqual(title, "[AI 技術] 2026-04-13 00:05")

    def test_notion_blocks_structure(self):
        sender = NotionSender(dry_run=True)
        blocks = sender.build_blocks(build_report(), "測試報告", "blue_background")

        heading_blocks = [block for block in blocks if block["type"] == "heading_3"]
        self.assertEqual(len(heading_blocks), 4)
        self.assertEqual(
            heading_blocks[0]["heading_3"]["rich_text"][0]["text"]["content"],
            "測試標題1",
        )

        callout_blocks = [block for block in blocks if block["type"] == "callout"]
        self.assertEqual(len(callout_blocks), 4)
        link_text = callout_blocks[0]["callout"]["rich_text"][1]["text"]["link"]["url"]
        self.assertEqual(link_text, "https://example.com/news1")
        metadata_lines = [
            block["paragraph"]["rich_text"][0]["text"]["content"]
            for block in blocks
            if block["type"] == "paragraph" and "來源:" in block["paragraph"]["rich_text"][0]["text"]["content"]
        ]
        self.assertTrue(any("來源: Example" in line for line in metadata_lines))

    def test_discord_payload_builds_without_live_send(self):
        sender = DiscordSender(dry_run=True)
        payload = sender.send_ai_tech_report(build_report(), notion_url=None)
        description = payload["embeds"][0]["description"]
        self.assertIn("測試標題1", description)
        self.assertIn("這是未來展望", description)
        self.assertIn("這是未來展望\n\n📎 其餘 1 則延伸內容與來源細節請看 Notion", description)
        self.assertIn("其餘 1 則延伸內容與來源細節請看 Notion", description)
        self.assertNotIn("測試標題4", description)

    def test_discord_stock_payload_formats_us_and_tw_differently(self):
        sender = DiscordSender(dry_run=True)
        payload = sender.send_stock_and_analysis(
            us_stocks=[{"symbol": "NVDA", "price": 188.63, "change": 5.61, "range": "184.3-190.0"}],
            tw_stocks=[
                {"symbol": "0050", "price": 80.75, "change": 1.7, "range": "79.9-80.8"},
                {"symbol": "2330", "price": 2000.0, "change": 60.0, "range": "1970.0-2000.0"},
            ],
            report=build_report(),
            notion_url=None,
        )
        description = payload["embeds"][0]["description"]
        self.assertIn("現:188.63 | 變:+5.61", description)
        self.assertIn("現:80.75 | 變:+1.70", description)
        self.assertIn("區:79.90-80.80", description)
        self.assertIn("現:2000 | 變:+60", description)
        self.assertIn("區:1970-2000", description)
