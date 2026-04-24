import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.deliverers.discord_sender import DiscordSender
from src.deliverers.notion_sender import NotionSender
from src.models import AnalyzedReport, ReportItem
from main import attach_ai_appendix, build_reports


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
        heading_titles = [
            block["heading_3"]["rich_text"][0]["text"]["content"] for block in heading_blocks
        ]
        self.assertEqual(heading_titles, ["測試標題1", "測試標題2", "測試標題3", "測試標題4"])
        self.assertTrue(
            all(
                not block["heading_3"]["rich_text"][0]["text"].get("link")
                for block in heading_blocks
            )
        )

        callout_blocks = [block for block in blocks if block["type"] == "callout"]
        self.assertEqual(len(callout_blocks), 4)
        self.assertTrue(
            callout_blocks[0]["callout"]["rich_text"][0]["text"]["content"].startswith("深度洞察: ")
        )
        self.assertFalse(
            callout_blocks[0]["callout"]["rich_text"][0]["text"]["content"].startswith("💡 ")
        )
        link_text = callout_blocks[0]["callout"]["rich_text"][1]["text"]["link"]["url"]
        self.assertEqual(link_text, "https://example.com/news1")
        metadata_lines = [
            block["paragraph"]["rich_text"][0]["text"]["content"]
            for block in blocks
            if block["type"] == "paragraph" and "來源:" in block["paragraph"]["rich_text"][0]["text"]["content"]
        ]
        self.assertTrue(any("來源: Example" in line for line in metadata_lines))

    def test_notion_blocks_include_appendix_items(self):
        sender = NotionSender(dry_run=True)
        report = build_report()
        attach_ai_appendix(
            report,
            [
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
                    title="附錄來源",
                    url="https://example.com/appendix",
                    summary="附錄摘要",
                    insight="附錄洞察",
                    source_name="Claude Code Changelog",
                    source_type="official_news",
                    published_at="2026-04-12T12:00:00Z",
                ),
            ],
        )

        blocks = sender.build_blocks(report, "測試報告", "blue_background")
        text_blobs = []
        for block in blocks:
            block_type = block["type"]
            payload = block[block_type]
            rich_text = payload.get("rich_text", [])
            text_blobs.append("".join(part.get("text", {}).get("content", "") for part in rich_text))

        self.assertTrue(any("附錄來源" in blob for blob in text_blobs))
        self.assertFalse(any("延伸來源附錄" in blob for blob in text_blobs))

    def test_notion_skips_invalid_links_but_keeps_insight_text(self):
        sender = NotionSender(dry_run=True)
        blocks = sender._build_item_blocks(
            "無效連結測試",
            "javascript:alert(1)",
            "摘要",
            "這是洞察",
            "blue_background",
            source_name="Example",
            source_type="news",
            published_at="2026-04-12T10:00:00Z",
        )

        callout = next(block for block in blocks if block["type"] == "callout")
        rich_text = callout["callout"]["rich_text"]
        self.assertEqual(len(rich_text), 1)
        self.assertEqual(rich_text[0]["text"]["content"], "深度洞察: 這是洞察\n\n")

    def test_notion_extra_items_sort_providers_then_community_then_github(self):
        sender = NotionSender(dry_run=True)
        items = [
            {
                "title": "[GitHub] repo",
                "url": "https://example.com/github",
                "desc": "Claude Code helper repo",
                "source_name": "GitHub",
                "source_type": "github_repo",
                "published_at": "2026-04-13T00:00:00Z",
            },
            {
                "title": "[Reddit] Ultraplan",
                "url": "https://example.com/reddit",
                "desc": "Ultraplan feels new",
                "source_name": "Reddit/ClaudeAI",
                "source_type": "community",
                "published_at": "2026-04-12T00:00:00Z",
            },
            {
                "title": "[Official] Gemini notebooks",
                "url": "https://example.com/notebooks",
                "desc": "NotebookLM in Gemini",
                "source_name": "Google Blog Gemini App",
                "source_type": "official_news",
                "published_at": "2026-04-11T00:00:00Z",
            },
        ]

        report = AnalyzedReport(
            title="AI",
            summary="summary",
            items=[],
            outlook="outlook",
            outlook_label="label",
            metadata={"appendix_items": items},
        )
        ordered_titles = [item["title"] for item in sender._sorted_render_items(report)]
        self.assertEqual(
            ordered_titles,
            ["[Official] Gemini notebooks", "[Reddit] Ultraplan", "[GitHub] repo"],
        )

    def test_notion_keeps_primary_items_before_appendix_items(self):
        sender = NotionSender(dry_run=True)
        report = build_report()
        report.metadata["appendix_items"] = [
            {
                "title": "[Official] Gemini notebooks",
                "url": "https://example.com/notebooks",
                "summary": "NotebookLM in Gemini",
                "insight": "官方產品更新。",
                "source_name": "Google Blog Gemini App",
                "source_type": "official_news",
                "published_at": "2026-04-13T00:00:00Z",
            },
            {
                "title": "[GitHub] repo",
                "url": "https://example.com/github",
                "summary": "Claude Code helper repo",
                "insight": "社群工具補充。",
                "source_name": "GitHub",
                "source_type": "github_repo",
                "published_at": "2026-04-12T00:00:00Z",
            },
        ]

        ordered_titles = [item["title"] for item in sender._sorted_render_items(report)]
        self.assertEqual(ordered_titles[:4], ["測試標題1", "測試標題2", "測試標題3", "測試標題4"])
        self.assertEqual(ordered_titles[4:], ["[Official] Gemini notebooks", "[GitHub] repo"])

    def test_notion_dedupes_primary_and_appendix_but_keeps_primary_item(self):
        sender = NotionSender(dry_run=True)
        report = AnalyzedReport(
            title="AI",
            summary="summary",
            items=[
                ReportItem(
                    title="[Official] Introducing GPT-5.5",
                    url="https://openai.com/index/introducing-gpt-5-5",
                    summary="Introducing GPT-5.5.",
                    insight="primary",
                    source_name="OpenAI News RSS",
                    source_type="official_news",
                    published_at="2026-04-23T11:00:00Z",
                )
            ],
            outlook="outlook",
            outlook_label="label",
            metadata={
                "appendix_items": [
                    {
                        "title": "[Official] GPT-5.5 System Card",
                        "url": "https://openai.com/index/gpt-5-5-system-card",
                        "summary": "Safety evaluations for GPT-5.5.",
                        "insight": "appendix",
                        "source_name": "OpenAI News RSS",
                        "source_type": "official_news",
                        "published_at": "2026-04-23T11:00:00Z",
                    },
                    {
                        "title": "[Official] Memory for Claude Managed Agents",
                        "url": "https://platform.claude.com/docs/en/managed-agents/memory",
                        "summary": "Claude Managed Agents memory beta.",
                        "insight": "appendix",
                        "source_name": "Claude Platform Release Notes",
                        "source_type": "official_news",
                        "published_at": "2026-04-23T00:00:00Z",
                    },
                ]
            },
        )

        ordered_titles = [item["title"] for item in sender._sorted_render_items(report)]
        self.assertEqual(ordered_titles.count("[Official] Introducing GPT-5.5"), 1)
        self.assertNotIn("[Official] GPT-5.5 System Card", ordered_titles)
        self.assertIn("[Official] Memory for Claude Managed Agents", ordered_titles)

    def test_notion_keeps_short_distinct_stock_titles_with_different_urls(self):
        sender = NotionSender(dry_run=True)
        report = AnalyzedReport(
            title="Stock",
            summary="summary",
            items=[
                ReportItem(
                    title="Tesla 增加資本支出",
                    url="https://techcrunch.com/tesla-capex",
                    summary="Tesla increased capex.",
                    insight="capex",
                    source_name="TechCrunch",
                    source_type="news",
                    published_at="2026-04-22T00:00:00Z",
                ),
                ReportItem(
                    title="Tesla財報公佈",
                    url="https://techcrunch.com/tesla-earnings",
                    summary="Tesla reported Q1 revenue.",
                    insight="earnings",
                    source_name="TechCrunch",
                    source_type="news",
                    published_at="2026-04-22T01:00:00Z",
                ),
            ],
            outlook="outlook",
            outlook_label="label",
        )

        ordered_titles = [item["title"] for item in sender._sorted_render_items(report)]
        self.assertIn("Tesla 增加資本支出", ordered_titles)
        self.assertIn("Tesla財報公佈", ordered_titles)

    def test_notion_appendix_cleans_html_noise(self):
        sender = NotionSender(dry_run=True)
        report = build_report()
        report.metadata["appendix_items"] = [
            {
                "title": "附錄來源",
                "url": "https://example.com/appendix",
                "desc": 'Search match for "ultraplan" | <!-- SC_OFF --><div class="md"><p>Rolling out now</p></div>',
                "source_name": "Reddit/ClaudeAI",
                "source_type": "community",
                "published_at": "2026-04-12T12:00:00Z",
            }
        ]

        blocks = sender.build_blocks(report, "測試報告", "blue_background")
        text_blobs = []
        for block in blocks:
            block_type = block["type"]
            payload = block[block_type]
            rich_text = payload.get("rich_text", [])
            text_blobs.append("".join(part.get("text", {}).get("content", "") for part in rich_text))

        self.assertTrue(any("Rolling out now" in blob for blob in text_blobs))
        self.assertFalse(any("SC_OFF" in blob for blob in text_blobs))
        self.assertFalse(any("Search match for" in blob for blob in text_blobs))

    def test_notion_blocks_cap_extra_items_before_hard_truncate(self):
        sender = NotionSender(dry_run=True)
        report = build_report()
        report.metadata["appendix_items"] = [
            {
                "title": f"附錄來源{i}",
                "url": f"https://example.com/appendix-{i}",
                "summary": "這是簡版中文摘要。",
                "insight": "",
                "source_name": "Example",
                "source_type": "official_news",
                "published_at": "2026-04-12T12:00:00Z",
            }
            for i in range(40)
        ]

        blocks = sender.build_blocks(report, "測試報告", "blue_background")
        self.assertLessEqual(len(blocks), 100)

    def test_discord_payload_builds_without_live_send(self):
        sender = DiscordSender(dry_run=True)
        payload = sender.send_ai_tech_report(build_report(), notion_url=None)
        description = payload["embeds"][0]["description"]
        self.assertIn("測試標題1", description)
        self.assertIn("這是未來展望", description)
        self.assertIn("這是未來展望\n\n📎 其餘 1 則延伸內容與來源細節請看 Notion", description)
        self.assertIn("其餘 1 則延伸內容與來源細節請看 Notion", description)
        self.assertNotIn("測試標題4", description)

    def test_discord_payload_dedupes_release_family_inside_top_items(self):
        sender = DiscordSender(dry_run=True)
        report = AnalyzedReport(
            title="AI 技術前沿情報",
            summary="summary",
            items=[
                ReportItem(
                    title="[Official] Introducing GPT-5.5",
                    url="https://openai.com/index/introducing-gpt-5-5",
                    summary="Introducing GPT-5.5.",
                    insight="primary",
                    source_name="OpenAI News RSS",
                    source_type="official_news",
                    published_at="2026-04-23T11:00:00Z",
                ),
                ReportItem(
                    title="[Official] GPT-5.5 System Card",
                    url="https://openai.com/index/gpt-5-5-system-card",
                    summary="Safety evaluations for GPT-5.5.",
                    insight="duplicate",
                    source_name="OpenAI News RSS",
                    source_type="official_news",
                    published_at="2026-04-23T11:00:00Z",
                ),
                ReportItem(
                    title="[Official] Memory for Claude Managed Agents",
                    url="https://platform.claude.com/docs/en/managed-agents/memory",
                    summary="Claude memory beta.",
                    insight="distinct",
                    source_name="Claude Platform Release Notes",
                    source_type="official_news",
                    published_at="2026-04-23T00:00:00Z",
                ),
            ],
            outlook="outlook",
            outlook_label="label",
        )

        payload = sender.send_ai_tech_report(report, notion_url=None)
        description = payload["embeds"][0]["description"]
        self.assertEqual(description.count("GPT-5.5"), 2)
        self.assertIn("Introducing GPT-5.5", description)
        self.assertNotIn("GPT-5.5 System Card", description)
        self.assertIn("Memory for Claude Managed Agents", description)

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

    @patch("main.select_ai_report_candidates")
    @patch("main.RunStateStore")
    @patch("main.DiscordSender")
    @patch("main.NotionSender")
    @patch("main.AIAnalyzer")
    def test_build_reports_stock_appendix_uses_recent_items_not_selected_pool(
        self,
        mock_analyzer_cls,
        mock_notion_cls,
        mock_discord_cls,
        mock_state_store_cls,
        mock_select_candidates,
    ):
        primary_stock_item = {
            "title": "Nvidia backed SiFive valuation climbs",
            "url": "https://example.com/sifive",
            "desc": "SiFive valuation climbed after new AI chip optimism.",
            "source_name": "TechCrunch",
            "source_type": "news",
            "published_at": "2026-04-11T00:00:00Z",
        }
        appendix_stock_item = {
            "title": "TSMC sees stronger AI demand into next quarter",
            "url": "https://example.com/tsmc-demand",
            "desc": "Foundry demand remains firm as AI infrastructure orders continue.",
            "source_name": "Reuters",
            "source_type": "news",
            "published_at": "2026-04-10T00:00:00Z",
        }

        mock_select_candidates.side_effect = lambda items, limit: items[:1]

        analyzer = mock_analyzer_cls.return_value
        analyzer.analyze_stock_market.return_value = AnalyzedReport(
            title="Stock",
            summary="stock summary",
            items=[
                ReportItem(
                    title=primary_stock_item["title"],
                    url=primary_stock_item["url"],
                    summary=primary_stock_item["desc"],
                    insight="市場正在重估 AI 晶片設計公司的成長空間。",
                    source_name=primary_stock_item["source_name"],
                    source_type=primary_stock_item["source_type"],
                    published_at=primary_stock_item["published_at"],
                )
            ],
            outlook="stock outlook",
            outlook_label="outlook",
        )
        analyzer.build_stock_brief_item.side_effect = lambda item: {
            "title": item["title"] if isinstance(item, dict) else item.title,
            "url": item["url"] if isinstance(item, dict) else item.url,
            "summary": item["desc"] if isinstance(item, dict) else item.desc,
            "insight": "後續可觀察供應鏈訂單與估值是否同步擴張。",
            "source_name": item["source_name"] if isinstance(item, dict) else item.source_name,
            "source_type": item["source_type"] if isinstance(item, dict) else item.source_type,
            "published_at": item["published_at"] if isinstance(item, dict) else item.published_at,
        }
        analyzer.analyze_ai_tech.return_value = AnalyzedReport(
            title="AI",
            summary="ai summary",
            items=[],
            outlook="ai outlook",
            outlook_label="outlook",
        )
        analyzer.build_ai_brief_item.side_effect = lambda item: {
            "title": item["title"] if isinstance(item, dict) else item.title,
            "url": item["url"] if isinstance(item, dict) else item.url,
            "summary": item["desc"] if isinstance(item, dict) else item.desc,
            "insight": "",
            "source_name": item["source_name"] if isinstance(item, dict) else item.source_name,
            "source_type": item["source_type"] if isinstance(item, dict) else item.source_type,
            "published_at": item["published_at"] if isinstance(item, dict) else item.published_at,
        }

        mock_notion_cls.return_value.create_stock_insight_report.return_value = None
        mock_notion_cls.return_value.create_ai_tech_report.return_value = None
        mock_discord_cls.return_value.send_stock_and_analysis.return_value = {}
        mock_discord_cls.return_value.send_ai_tech_report.return_value = {}

        state_store = mock_state_store_cls.return_value
        state_store.filter_new_items.side_effect = lambda category, items, limit: (items[:limit], 0)

        fixed_now = datetime(2026, 4, 12, 0, 0, 0, tzinfo=timezone.utc)
        result = build_reports(
            {
                "us_stocks": [],
                "tw_stocks": [],
                "stock_news": [primary_stock_item, appendix_stock_item],
                "ai_news": [],
            },
            enable_ai=False,
            dry_run=True,
            now=fixed_now,
        )

        appendix_urls = {
            item["url"] for item in result["stock_report"].metadata.get("appendix_items", [])
        }
        self.assertIn("https://example.com/tsmc-demand", appendix_urls)

    @patch("main.select_ai_report_candidates")
    @patch("main.RunStateStore")
    @patch("main.DiscordSender")
    @patch("main.NotionSender")
    @patch("main.AIAnalyzer")
    def test_build_reports_appendix_uses_recent_items_not_selected_pool(
        self,
        mock_analyzer_cls,
        mock_notion_cls,
        mock_discord_cls,
        mock_state_store_cls,
        mock_select_candidates,
    ):
        selected_item = {
            "title": "[Official] OpenAI ships a new Responses API update",
            "url": "https://example.com/openai",
            "desc": "Responses API adds a new tool",
            "source_name": "OpenAI API Changelog",
            "source_type": "official_news",
            "published_at": "2026-04-10T00:00:00Z",
        }
        appendix_only_item = {
            "title": "[Official] Try notebooks in Gemini to easily keep track of projects",
            "url": "https://example.com/notebooks",
            "desc": "NotebookLM and Gemini app project workflow",
            "source_name": "Google Blog Gemini App",
            "source_type": "official_news",
            "published_at": "2026-04-08T00:00:00Z",
        }

        mock_select_candidates.side_effect = lambda items, limit: items[:1]

        analyzer = mock_analyzer_cls.return_value
        analyzer.analyze_stock_market.return_value = AnalyzedReport(
            title="Stock",
            summary="stock summary",
            items=[],
            outlook="stock outlook",
            outlook_label="outlook",
        )
        analyzer.analyze_ai_tech.return_value = AnalyzedReport(
            title="AI",
            summary="ai summary",
            items=[
                ReportItem(
                    title=selected_item["title"],
                    url=selected_item["url"],
                    summary=selected_item["desc"],
                    insight="selected",
                    source_name=selected_item["source_name"],
                    source_type=selected_item["source_type"],
                    published_at=selected_item["published_at"],
                )
            ],
            outlook="ai outlook",
            outlook_label="outlook",
        )
        analyzer.build_ai_brief_item.side_effect = lambda item: {
            "title": item["title"] if isinstance(item, dict) else item.title,
            "url": item["url"] if isinstance(item, dict) else item.url,
            "summary": item["desc"] if isinstance(item, dict) else item.desc,
            "insight": "",
            "source_name": item["source_name"] if isinstance(item, dict) else item.source_name,
            "source_type": item["source_type"] if isinstance(item, dict) else item.source_type,
            "published_at": item["published_at"] if isinstance(item, dict) else item.published_at,
        }

        mock_notion_cls.return_value.create_stock_insight_report.return_value = None
        mock_notion_cls.return_value.create_ai_tech_report.return_value = None
        mock_discord_cls.return_value.send_stock_and_analysis.return_value = {}
        mock_discord_cls.return_value.send_ai_tech_report.return_value = {}

        state_store = mock_state_store_cls.return_value
        state_store.filter_new_items.side_effect = lambda category, items, limit: (items[:limit], 0)

        fixed_now = datetime(2026, 4, 12, 0, 0, 0, tzinfo=timezone.utc)
        result = build_reports(
            {
                "us_stocks": [],
                "tw_stocks": [],
                "stock_news": [],
                "ai_news": [selected_item, appendix_only_item],
            },
            enable_ai=False,
            dry_run=True,
            now=fixed_now,
        )

        appendix_urls = {
            item["url"] for item in result["ai_report"].metadata.get("appendix_items", [])
        }
        self.assertIn("https://example.com/notebooks", appendix_urls)

    @patch("main.select_ai_report_candidates")
    @patch("main.RunStateStore")
    @patch("main.DiscordSender")
    @patch("main.NotionSender")
    @patch("main.AIAnalyzer")
    def test_build_reports_ai_appendix_keeps_older_high_impact_mythos_item(
        self,
        mock_analyzer_cls,
        mock_notion_cls,
        mock_discord_cls,
        mock_state_store_cls,
        mock_select_candidates,
    ):
        selected_item = {
            "title": "[Official] OpenAI ships a new Responses API update",
            "url": "https://example.com/openai",
            "desc": "Responses API adds a new tool",
            "source_name": "OpenAI API Changelog",
            "source_type": "official_news",
            "published_at": "2026-04-24T00:00:00Z",
        }
        older_mythos_item = {
            "title": "Anthropic investigates unauthorized access to Claude Mythos Preview",
            "url": "https://example.com/claude-mythos-breach",
            "desc": "Project Glasswing frontier cybersecurity model access may have been exposed.",
            "source_name": "CBS News",
            "source_type": "news",
            "published_at": "2026-04-07T00:00:00Z",
        }

        mock_select_candidates.side_effect = lambda items, limit: items[:1]

        analyzer = mock_analyzer_cls.return_value
        analyzer.analyze_stock_market.return_value = AnalyzedReport(
            title="Stock",
            summary="stock summary",
            items=[],
            outlook="stock outlook",
            outlook_label="outlook",
        )
        analyzer.analyze_ai_tech.return_value = AnalyzedReport(
            title="AI",
            summary="ai summary",
            items=[
                ReportItem(
                    title=selected_item["title"],
                    url=selected_item["url"],
                    summary=selected_item["desc"],
                    insight="selected",
                    source_name=selected_item["source_name"],
                    source_type=selected_item["source_type"],
                    published_at=selected_item["published_at"],
                )
            ],
            outlook="ai outlook",
            outlook_label="outlook",
        )
        analyzer.build_ai_brief_item.side_effect = lambda item: {
            "title": item["title"] if isinstance(item, dict) else item.title,
            "url": item["url"] if isinstance(item, dict) else item.url,
            "summary": item["desc"] if isinstance(item, dict) else item.desc,
            "insight": "",
            "source_name": item["source_name"] if isinstance(item, dict) else item.source_name,
            "source_type": item["source_type"] if isinstance(item, dict) else item.source_type,
            "published_at": item["published_at"] if isinstance(item, dict) else item.published_at,
        }

        mock_notion_cls.return_value.create_stock_insight_report.return_value = None
        mock_notion_cls.return_value.create_ai_tech_report.return_value = None
        mock_discord_cls.return_value.send_stock_and_analysis.return_value = {}
        mock_discord_cls.return_value.send_ai_tech_report.return_value = {}

        state_store = mock_state_store_cls.return_value
        state_store.filter_new_items.side_effect = lambda category, items, limit: (items[:limit], 0)

        result = build_reports(
            {
                "us_stocks": [],
                "tw_stocks": [],
                "stock_news": [],
                "ai_news": [selected_item, older_mythos_item],
            },
            enable_ai=False,
            dry_run=True,
            now=datetime(2026, 4, 24, 0, 0, 0, tzinfo=timezone.utc),
        )

        report_urls = {item["url"] for item in result["ai_items"]}
        appendix_urls = {
            item["url"] for item in result["ai_report"].metadata.get("appendix_items", [])
        }
        self.assertNotIn("https://example.com/claude-mythos-breach", report_urls)
        self.assertIn("https://example.com/claude-mythos-breach", appendix_urls)

    @patch("main.select_ai_report_candidates")
    @patch("main.RunStateStore")
    @patch("main.DiscordSender")
    @patch("main.NotionSender")
    @patch("main.AIAnalyzer")
    def test_build_reports_appendix_skips_generic_community_fallback_items(
        self,
        mock_analyzer_cls,
        mock_notion_cls,
        mock_discord_cls,
        mock_state_store_cls,
        mock_select_candidates,
    ):
        selected_item = {
            "title": "[Official] Managed Agents overview",
            "url": "https://example.com/managed-agents",
            "desc": "Managed Agents overview",
            "source_name": "Claude Platform Release Notes",
            "source_type": "official_news",
            "published_at": "2026-04-10T00:00:00Z",
        }
        noisy_reddit = {
            "title": "[Reddit r/ClaudeAI] the golden age is over",
            "url": "https://example.com/reddit-noise",
            "desc": "RSS fallback | <div>generic discussion</div>",
            "source_name": "Reddit/ClaudeAI",
            "source_type": "community",
            "published_at": "2026-04-09T00:00:00Z",
            "metadata": {},
        }

        mock_select_candidates.side_effect = lambda items, limit: items[:1]

        analyzer = mock_analyzer_cls.return_value
        analyzer.analyze_stock_market.return_value = AnalyzedReport(
            title="Stock",
            summary="stock summary",
            items=[],
            outlook="stock outlook",
            outlook_label="outlook",
        )
        analyzer.analyze_ai_tech.return_value = AnalyzedReport(
            title="AI",
            summary="ai summary",
            items=[
                ReportItem(
                    title=selected_item["title"],
                    url=selected_item["url"],
                    summary=selected_item["desc"],
                    insight="selected",
                    source_name=selected_item["source_name"],
                    source_type=selected_item["source_type"],
                    published_at=selected_item["published_at"],
                )
            ],
            outlook="ai outlook",
            outlook_label="outlook",
        )
        analyzer.build_ai_brief_item.side_effect = lambda item: {
            "title": item["title"] if isinstance(item, dict) else item.title,
            "url": item["url"] if isinstance(item, dict) else item.url,
            "summary": item["desc"] if isinstance(item, dict) else item.desc,
            "insight": "",
            "source_name": item["source_name"] if isinstance(item, dict) else item.source_name,
            "source_type": item["source_type"] if isinstance(item, dict) else item.source_type,
            "published_at": item["published_at"] if isinstance(item, dict) else item.published_at,
        }

        mock_notion_cls.return_value.create_stock_insight_report.return_value = None
        mock_notion_cls.return_value.create_ai_tech_report.return_value = None
        mock_discord_cls.return_value.send_stock_and_analysis.return_value = {}
        mock_discord_cls.return_value.send_ai_tech_report.return_value = {}

        state_store = mock_state_store_cls.return_value
        state_store.filter_new_items.side_effect = lambda category, items, limit: (items[:limit], 0)

        result = build_reports(
            {
                "us_stocks": [],
                "tw_stocks": [],
                "stock_news": [],
                "ai_news": [selected_item, noisy_reddit],
            },
            enable_ai=False,
            dry_run=True,
        )

        appendix_urls = {
            item["url"] for item in result["ai_report"].metadata.get("appendix_items", [])
        }
        self.assertNotIn("https://example.com/reddit-noise", appendix_urls)
