from __future__ import annotations

import re
from html import unescape
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlsplit, urlunsplit

from src.config import Config
from src.models import AnalyzedReport
from src.utils.logger import logger

try:
    from notion_client import Client
except ImportError:  # pragma: no cover - optional dependency in dry-run
    Client = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency in minimal test envs
    BeautifulSoup = None


class NotionSender:
    _MAX_BLOCKS = 100
    _PROVIDER_KEYWORDS = (
        "claude",
        "anthropic",
        "gpt",
        "chatgpt",
        "openai",
        "codex",
        "gemini",
        "xai",
        "grok",
    )

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

        reserved_tail_blocks = 2 if report.outlook else 0
        for item in self._sorted_render_items(report):
            item_blocks = self._build_item_blocks(
                item["title"],
                item["url"],
                item["summary"],
                item["insight"],
                bg_color,
                source_name=item["source_name"],
                source_type=item["source_type"],
                published_at=item["published_at"],
            )
            if len(blocks) + len(item_blocks) + reserved_tail_blocks > self._MAX_BLOCKS:
                break
            blocks.extend(item_blocks)

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
        normalized_url = self._normalize_link_url(url)
        blocks: list[dict] = [
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [
                        {
                            "text": {
                                "content": title,
                            }
                        }
                    ]
                },
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
        if insight or normalized_url:
            prefix = f"深度洞察: {insight}\n\n" if insight else ""
            rich_text = [{"text": {"content": prefix}}] if prefix else []
            if normalized_url:
                rich_text.append(
                    {
                        "text": {"content": "🔗 查看原文", "link": {"url": normalized_url}},
                        "annotations": {
                            "italic": True,
                            "bold": True,
                            "color": "blue",
                        },
                    }
                )
            blocks.append(
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": rich_text,
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

        title = self._build_title(title_prefix)
        page = self.notion.pages.create(
            parent={"database_id": Config.NOTION_PAGE_ID},
            properties={"Name": {"title": [{"text": {"content": title}}]}},
            children=self._cap_blocks(blocks),
        )
        logger.info("Detailed report created in Notion.")
        return page["url"]

    def _clean_appendix_snippet(self, text: str) -> str:
        if not text:
            return ""

        cleaned = unescape(text)
        cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
        if BeautifulSoup is not None:
            cleaned = BeautifulSoup(cleaned, "html.parser").get_text(" ", strip=True)
        else:
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        cleaned = re.sub(
            r"^(RSS fallback|Search match for [^|]+|Collected from [^.]+ listing page\.)\s*\|?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _sorted_render_items(self, report: AnalyzedReport) -> list[dict]:
        primary_items: list[dict] = []
        for item in report.items:
            primary_items.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "summary": item.summary,
                    "insight": item.insight,
                    "source_name": item.source_name,
                    "source_type": item.source_type,
                    "published_at": item.published_at,
                    "is_primary": True,
                }
            )

        appendix_items: list[dict] = []
        for item in report.metadata.get("appendix_items", []):
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            if not title or not url:
                continue
            snippet = self._clean_appendix_snippet(str(item.get("summary") or item.get("desc", "")).strip())
            if len(snippet) > 180:
                snippet = snippet[:177].rstrip() + "..."
            appendix_items.append(
                {
                    "title": title,
                    "url": url,
                    "summary": snippet,
                    "insight": str(item.get("insight", "")).strip(),
                    "source_name": str(item.get("source_name", "")).strip(),
                    "source_type": str(item.get("source_type", "")).strip(),
                    "published_at": str(item.get("published_at", "")).strip() or None,
                    "is_primary": False,
                }
            )

        appendix_items.sort(key=self._render_item_sort_key)
        return primary_items + appendix_items

    def _render_item_sort_key(self, item: dict) -> tuple[int, int, int, float]:
        source_type = str(item.get("source_type", "")).strip().lower()
        text = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("summary", "")),
                str(item.get("source_name", "")),
            ]
        ).lower()
        has_provider = any(keyword in text for keyword in self._PROVIDER_KEYWORDS)

        if source_type in {"github_repo", "github_release"}:
            bucket = 4
        elif source_type == "community" and has_provider:
            bucket = 2
        elif source_type == "community":
            bucket = 3
        elif has_provider:
            bucket = 0
        else:
            bucket = 1

        published_at = self._parse_sort_datetime(str(item.get("published_at", "")).strip())
        published_key = -published_at.timestamp() if published_at else float("inf")
        primary_penalty = 0 if item.get("is_primary") else 1
        return (bucket, 0 if has_provider else 1, primary_penalty, published_key)

    def _parse_sort_datetime(self, value: str) -> datetime | None:
        cleaned = (value or "").strip()
        if not cleaned:
            return None
        if cleaned.endswith("Z"):
            cleaned = f"{cleaned[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _build_title(self, title_prefix: str, now: datetime | None = None) -> str:
        tw_tz = timezone(timedelta(hours=8))
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return f"{title_prefix} {current.astimezone(tw_tz).strftime('%Y-%m-%d %H:%M')}"

    def _normalize_link_url(self, value: str) -> str | None:
        raw = (value or "").strip()
        if not raw:
            return None
        cleaned = "".join(ch for ch in raw if ch.isprintable() and ch not in "\r\n\t")
        parsed = urlsplit(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            logger.warning("Skipping Notion link with unsupported URL: %r", value)
            return None
        normalized = parsed._replace(
            path=quote(parsed.path, safe="/%:@-._~!$&'()*+,;="),
            query=quote(parsed.query, safe="=&%:@-._~!$'()*+,;/?"),
            fragment=quote(parsed.fragment, safe="=%:@-._~!$&'()*+,;/?"),
        )
        return urlunsplit(normalized)

    def _cap_blocks(self, blocks: list[dict], limit: int = 100) -> list[dict]:
        if len(blocks) <= limit:
            return blocks
        logger.warning("Notion blocks exceeded limit; truncating from %s to %s.", len(blocks), limit)
        return blocks[:limit]
