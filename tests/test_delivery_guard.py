import unittest
from unittest.mock import Mock, patch

from src.config import Config
from src.deliverers.discord_sender import DiscordSender
from src.deliverers.guard import should_deliver
from src.deliverers.notion_sender import NotionSender
from src.models import AnalyzedReport, ReportItem


def _sample_report() -> AnalyzedReport:
    return AnalyzedReport(
        title="AI 技術前沿情報",
        summary="摘要",
        items=[
            ReportItem(
                title="測試標題",
                url="https://example.com/item",
                summary="s",
                insight="i",
                source_name="Example",
                source_type="official_news",
                published_at="2026-04-12T10:00:00Z",
            )
        ],
        outlook="o",
        outlook_label="🔮 未來展望",
    )


class TestShouldDeliver(unittest.TestCase):
    def test_only_live_and_enabled_allows_delivery(self):
        cases = [
            (True, True, False),
            (True, False, False),
            (False, True, True),
            (False, False, False),
        ]
        for dry_run, enabled, expected in cases:
            with self.subTest(dry_run=dry_run, enabled=enabled):
                self.assertEqual(should_deliver(dry_run=dry_run, enabled=enabled), expected)


class TestDiscordSenderGuard(unittest.TestCase):
    def test_dry_run_skips_post(self):
        original_webhook = Config.DISCORD_WEBHOOK_URL
        try:
            # Webhook must be truthy here: it isolates the assertion to the
            # should_deliver() guard rather than the downstream missing-webhook check.
            Config.DISCORD_WEBHOOK_URL = "https://discord.example.com/webhook"
            session = Mock()
            sender = DiscordSender(dry_run=True, enabled=True, session=session)

            sender._deliver({"embeds": []})

            session.post.assert_not_called()
        finally:
            Config.DISCORD_WEBHOOK_URL = original_webhook

    def test_live_and_enabled_sends_post_with_payload(self):
        original_webhook = Config.DISCORD_WEBHOOK_URL
        try:
            Config.DISCORD_WEBHOOK_URL = "https://discord.example.com/webhook"
            session = Mock()
            session.post.return_value = Mock(raise_for_status=Mock())
            sender = DiscordSender(dry_run=False, enabled=True, session=session)
            payload = {"embeds": [{"title": "t"}]}

            sender._deliver(payload)

            session.post.assert_called_once()
            call = session.post.call_args
            self.assertEqual(call.args[0], Config.DISCORD_WEBHOOK_URL)
            self.assertEqual(call.kwargs["json"], payload)
        finally:
            Config.DISCORD_WEBHOOK_URL = original_webhook


class TestNotionSenderGuard(unittest.TestCase):
    def test_dry_run_skips_client_construction_and_create(self):
        original_token = Config.NOTION_TOKEN
        original_page_id = Config.NOTION_PAGE_ID
        try:
            Config.NOTION_TOKEN = "secret_token"
            Config.NOTION_PAGE_ID = "db-id"
            with patch("src.deliverers.notion_sender.Client") as client_cls:
                sender = NotionSender(dry_run=True, enabled=True)

                self.assertIsNone(sender.notion)
                client_cls.assert_not_called()

                url = sender._create_report(
                    _sample_report(),
                    title_prefix="[AI 技術]",
                    heading="h",
                    bg_color="gray_background",
                )

                self.assertIsNone(url)
                client_cls.return_value.pages.create.assert_not_called()
        finally:
            Config.NOTION_TOKEN = original_token
            Config.NOTION_PAGE_ID = original_page_id

    def test_live_and_enabled_constructs_client_and_creates_page(self):
        original_token = Config.NOTION_TOKEN
        original_page_id = Config.NOTION_PAGE_ID
        try:
            Config.NOTION_TOKEN = "secret_token"
            Config.NOTION_PAGE_ID = "db-id"
            with patch("src.deliverers.notion_sender.Client") as client_cls:
                client_instance = client_cls.return_value
                client_instance.pages.create.return_value = {"url": "https://notion.so/page"}

                sender = NotionSender(dry_run=False, enabled=True)

                client_cls.assert_called_once_with(auth="secret_token")
                self.assertIsNotNone(sender.notion)

                url = sender._create_report(
                    _sample_report(),
                    title_prefix="[AI 技術]",
                    heading="h",
                    bg_color="gray_background",
                )

                client_instance.pages.create.assert_called_once()
                self.assertEqual(url, "https://notion.so/page")
        finally:
            Config.NOTION_TOKEN = original_token
            Config.NOTION_PAGE_ID = original_page_id


if __name__ == "__main__":
    unittest.main()
