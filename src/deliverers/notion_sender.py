from __future__ import annotations

from datetime import datetime

from src.config import Config
from src.models import AnalyzedReport
from src.utils.logger import logger

try:
    from notion_client import Client
except ImportError:  # pragma: no cover - optional dependency in dry-run
    Client = None


class NotionSender:
    def __init__(
        self,
        *,
        dry_run: bool | None = None,
        enabled: bool | None = None,
    ):
        self.dry_run = Config.DRY_RUN if dry_run is None else dry_run
        self.enabled = Config.ENABLE_NOTION_DELIVERY if enabled is None else enabled
        self.notion = (
            Client(auth=Config.NOTION_TOKEN)
            if Client and Config.NOTION_TOKEN and self.enabled and not self.dry_run
            else None
        )

    def create_stock_insight_report(self, report: AnalyzedReport) -> str | None:
        return self._create_report(
            report,
            title_prefix="[投資情報]",
            heading="投資與產業分析報告",
            bg_color="blue_background",
        )

    def create_ai_tech_report(self, report: AnalyzedReport) -> str | None:
        return self._create_report(
            report,
            title_prefix="[AI 技術]",
            heading="AI 前沿技術觀察報告",
            bg_color="gray_background",
        )

    def build_blocks(self, report: AnalyzedReport, main_heading: str, bg_color: str) -> list[dict]:
        blocks: list[dict] = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"text": {"content": main_heading}}],
                    "color": bg_color,
                },
            },
            {"object": "block", "type": "divider", "divider": {}},
        ]

        if report.summary:
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": report.summary}}],
                        "color": "gray",
                    },
                }
            )

        for item in report.items:
            blocks.extend(
                self._build_item_blocks(
                    item.title,
                    item.url,
                    item.summary,
                    item.insight,
                    bg_color,
                    source_name=item.source_name,
                    source_type=item.source_type,
                    published_at=item.published_at,
                )
            )

        if report.outlook:
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            blocks.append(
                {
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": f"{report.outlook_label}: {report.outlook}",
                                }
                            }
                        ]
                    },
                }
            )
        return blocks

    def _build_item_blocks(
        self,
        title: str,
        url: str,
        summary: str,
        insight: str,
        bg_color: str,
        *,
        source_name: str,
        source_type: str,
        published_at: str | None,
    ) -> list[dict]:
        blocks: list[dict] = [
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"text": {"content": title}}]},
            }
        ]
        meta_parts = [
            part
            for part in [
                f"來源: {source_name}" if source_name and source_name != "unknown" else "",
                f"分類: {source_type}" if source_type and source_type != "unknown" else "",
                f"時間: {published_at}" if published_at else "",
            ]
            if part
        ]
        if meta_parts:
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": " | ".join(meta_parts)}}],
                        "color": "gray",
                    },
                }
            )
        if summary:
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"text": {"content": summary}}]},
                }
            )
        if insight:
            blocks.append(
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [
                            {"text": {"content": f"💡 深度洞察: {insight}\n\n"}},
                            {
                                "text": {"content": "🔗 查看原文", "link": {"url": url}},
                                "annotations": {
                                    "italic": True,
                                    "bold": True,
                                    "color": "blue",
                                },
                            },
                        ],
                        "icon": {"emoji": "💡"},
                        "color": "blue_background" if "blue" in bg_color else "gray_background",
                    },
                }
            )
        return blocks

    def _create_report(
        self,
        report: AnalyzedReport,
        *,
        title_prefix: str,
        heading: str,
        bg_color: str,
    ) -> str | None:
        blocks = self.build_blocks(report, heading, bg_color)
        if self.dry_run or not self.enabled:
            logger.info(
                "Notion delivery skipped (dry_run=%s, enabled=%s, blocks=%s).",
                self.dry_run,
                self.enabled,
                len(blocks),
            )
            return None
        if not self.notion or not Config.NOTION_PAGE_ID:
            logger.warning("Notion client or database id missing, skipping send.")
            return None

        title = f"{title_prefix} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        page = self.notion.pages.create(
            parent={"database_id": Config.NOTION_PAGE_ID},
            properties={"Name": {"title": [{"text": {"content": title}}]}},
            children=self._cap_blocks(blocks),
        )
        logger.info("Detailed report created in Notion.")
        return page["url"]

    def _cap_blocks(self, blocks: list[dict], limit: int = 100) -> list[dict]:
        if len(blocks) <= limit:
            return blocks
        logger.warning("Notion blocks exceeded limit; truncating from %s to %s.", len(blocks), limit)
        return blocks[:limit]
