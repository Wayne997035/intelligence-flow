"""One broken step must never stop the rest of the run, and a run where
everything degrades must still deliver something."""
import unittest
from unittest.mock import MagicMock, patch

import requests

import main
from src.config import Config
from src.deliverers.discord_sender import DiscordSender
from src.deliverers.notion_sender import NotionSender


def _fixture_inputs():
    bundle = main.load_fixture_bundle(main.DEFAULT_FIXTURE)
    bundle["_fixture"] = True
    return bundle


class _Response:
    def __init__(self, status):
        self.status_code = status
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return {}


class ResilienceTestBase(unittest.TestCase):
    def setUp(self):
        patches = [
            patch.object(Config, "WRITE_ARTIFACTS", False),
            patch.object(Config, "DISCORD_WEBHOOK_URL", "https://discord.test/webhook"),
            patch.object(Config, "ENABLE_DISCORD_DELIVERY", True),
            patch.object(Config, "ENABLE_NOTION_DELIVERY", True),
            patch.object(Config, "NOTION_PAGE_ID", "page"),
            patch.object(Config, "NOTION_TOKEN", "token"),
            patch("src.deliverers.discord_sender.time.sleep"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.discord_session = MagicMock()
        self.discord_session.post.return_value = _Response(204)
        self.notion_client = MagicMock()
        self.notion_client.pages.create.return_value = {"url": "https://notion.test/page"}

        def make_discord(dry_run=None, enabled=None):
            return DiscordSender(dry_run=dry_run, enabled=enabled, session=self.discord_session)

        def make_notion(dry_run=None, enabled=None):
            sender = NotionSender(dry_run=dry_run, enabled=enabled)
            sender.notion = self.notion_client
            return sender

        for target, factory in (("DiscordSender", make_discord), ("NotionSender", make_notion)):
            p = patch.object(main, target, side_effect=factory)
            p.start()
            self.addCleanup(p.stop)

    def run_live(self, inputs=None):
        return main.build_reports(inputs or _fixture_inputs(), enable_ai=False, dry_run=False)

    def sent_titles(self):
        return [call.kwargs["json"]["embeds"][0]["title"] for call in self.discord_session.post.call_args_list]


class TestDeliveryIsolation(ResilienceTestBase):
    def test_happy_path_sends_both_reports(self):
        result = self.run_live()
        self.assertEqual(self.sent_titles(), ["投資情報報告", "AI 技術前沿情報"])
        self.assertEqual(result["meta"]["errors"], {})

    def test_notion_outage_still_sends_discord(self):
        self.notion_client.pages.create.side_effect = Exception("Notion 502")
        result = self.run_live()
        self.assertEqual(self.sent_titles(), ["投資情報報告", "AI 技術前沿情報"])
        self.assertEqual(len(result["meta"]["errors"]["delivery"]), 2)
        self.assertNotIn("notion.test", result["stock_payload"]["embeds"][0]["description"])

    def test_discord_transient_error_is_retried(self):
        self.discord_session.post.side_effect = [_Response(500), _Response(204), _Response(204)]
        result = self.run_live()
        self.assertEqual(self.discord_session.post.call_count, 3)
        self.assertEqual(result["meta"]["errors"], {})

    def test_discord_failure_on_first_report_does_not_block_second(self):
        self.discord_session.post.side_effect = [_Response(500), _Response(500), _Response(204)]
        result = self.run_live()
        self.assertEqual(self.sent_titles()[-1], "AI 技術前沿情報")
        self.assertEqual(len(result["meta"]["errors"]["delivery"]), 1)
        self.assertIn("HTTP 500", result["meta"]["errors"]["delivery"][0])

    def test_discord_client_error_is_not_retried(self):
        self.discord_session.post.side_effect = [_Response(400), _Response(204)]
        self.run_live()
        self.assertEqual(self.discord_session.post.call_count, 2)


class TestReportIsolation(ResilienceTestBase):
    def test_analysis_crash_falls_back_to_rule_based_report(self):
        with patch.object(main.AIAnalyzer, "analyze_stock_market", side_effect=RuntimeError("boom")):
            result = self.run_live()
        self.assertEqual(self.sent_titles(), ["投資情報報告", "AI 技術前沿情報"])
        self.assertIn("stock analysis: RuntimeError: boom", result["meta"]["errors"]["report"])
        self.assertEqual(result["stock_report"].metadata.get("mode"), "fallback")

    def test_whole_section_failure_sends_notice_and_other_report(self):
        with patch.object(main, "balance_stock_news", side_effect=RuntimeError("ranking bug")):
            result = self.run_live()
        self.assertEqual(self.sent_titles(), ["⚠️ 投資情報報告產生失敗", "AI 技術前沿情報"])
        self.assertEqual(result["stock_report"].metadata["mode"], "failed")
        self.assertTrue(any("stock section" in e for e in result["meta"]["errors"]["report"]))

    def test_both_sections_failing_still_sends_two_notices(self):
        with patch.object(main, "balance_stock_news", side_effect=RuntimeError("a")), patch.object(
            main, "select_ai_report_candidates", side_effect=RuntimeError("b")
        ):
            self.run_live()
        self.assertEqual(self.sent_titles(), ["⚠️ 投資情報報告產生失敗", "⚠️ AI 技術前沿情報產生失敗"])

    def test_no_data_at_all_still_sends_reports(self):
        self.run_live({"us_stocks": [], "tw_stocks": [], "stock_news": [], "ai_news": []})
        self.assertEqual(self.sent_titles(), ["投資情報報告", "AI 技術前沿情報"])

    def test_state_store_failure_is_recorded_not_raised(self):
        with patch.object(main.RunStateStore, "save", side_effect=OSError("disk full")):
            result = self.run_live()
        self.assertEqual(len(self.sent_titles()), 2)
        self.assertIn("state save: OSError: disk full", result["meta"]["errors"]["state"])


class TestCollectionIsolation(unittest.TestCase):
    def _patch_collectors(self):
        fake_classes = {}
        for name in (
            "StockCollector",
            "NewsCollector",
            "TechCollector",
            "HFCollector",
            "ArxivCollector",
            "OfficialAICollector",
            "GitHubReleaseCollector",
            "GoogleNewsCollector",
        ):
            instance = MagicMock()
            for method in (
                "fetch_us_stocks", "fetch_tw_stocks", "fetch_ticker_news", "fetch_stock_news", "fetch_ai_tech_news",
                "fetch_stock_topics", "fetch_ai_topics", "fetch_updates", "fetch_latest_releases",
                "fetch_all_community_ai", "fetch_all_hf", "fetch_all_arxiv",
            ):
                getattr(instance, method).return_value = [{"title": f"{name}.{method}", "url": "https://x.test"}]
            fake_classes[name] = MagicMock(return_value=instance)
        for name, cls in fake_classes.items():
            p = patch.object(main, name, cls)
            p.start()
            self.addCleanup(p.stop)
        return fake_classes

    def test_one_broken_source_does_not_stop_collection(self):
        fake_classes = self._patch_collectors()
        fake_classes["NewsCollector"].return_value.fetch_stock_news.side_effect = ConnectionError("NewsAPI down")
        fake_classes["OfficialAICollector"].return_value.fetch_updates.side_effect = ValueError("bad HTML")

        inputs = main.collect_inputs(use_fixture=False)

        self.assertEqual(inputs["_source_counts"]["stock_news"]["newsapi"], 0)
        self.assertEqual(inputs["_source_counts"]["stock_news"]["ticker_news"], 1)
        self.assertEqual(inputs["_source_counts"]["ai_news"]["official"], 0)
        self.assertEqual(inputs["_source_counts"]["ai_news"]["arxiv"], 1)
        self.assertEqual(len(inputs["us_stocks"]), 1)
        self.assertEqual(len(inputs["_errors"]), 2)

    def test_collector_construction_failure_only_loses_its_own_sources(self):
        fake_classes = self._patch_collectors()
        fake_classes["StockCollector"].side_effect = RuntimeError("init")

        inputs = main.collect_inputs(use_fixture=False)

        self.assertEqual(inputs["us_stocks"], [])
        self.assertEqual(inputs["_source_counts"]["stock_news"]["ticker_news"], 0)
        self.assertEqual(inputs["_source_counts"]["stock_news"]["newsapi"], 1)
        self.assertEqual(len(inputs["ai_news"]), 7)
        self.assertIn("collector setup stock: RuntimeError: init", inputs["_errors"])


if __name__ == "__main__":
    unittest.main()
