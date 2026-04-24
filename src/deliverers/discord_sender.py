from __future__ import annotations

import requests

from src.config import Config
from src.models import AnalyzedReport
from src.pipeline import content_dedupe_key
from src.utils.logger import logger


class DiscordSender:
    DISCORD_ITEM_LIMIT = 3
    SUMMARY_LIMIT = 90
    INSIGHT_LIMIT = 110
    OUTLOOK_LIMIT = 1200

    def __init__(
        self,
        *,
        dry_run: bool | None = None,
        enabled: bool | None = None,
        session: requests.Session | None = None,
    ):
        self.dry_run = Config.DRY_RUN if dry_run is None else dry_run
        self.enabled = Config.ENABLE_DISCORD_DELIVERY if enabled is None else enabled
        self.session = session or requests.Session()

    def send_stock_and_analysis(
        self,
        us_stocks: list[dict],
        tw_stocks: list[dict],
        report: AnalyzedReport,
        notion_url: str | None,
    ) -> dict:
        payload = self._build_stock_payload(us_stocks, tw_stocks, report, notion_url)
        self._deliver(payload)
        return payload

    def send_ai_tech_report(
        self,
        report: AnalyzedReport,
        notion_url: str | None,
    ) -> dict:
        payload = self._build_ai_payload(report, notion_url)
        self._deliver(payload)
        return payload

    def _build_stock_payload(
        self,
        us_stocks: list[dict],
        tw_stocks: list[dict],
        report: AnalyzedReport,
        notion_url: str | None,
    ) -> dict:
        sections = [
            "🇺🇸 **美股**",
            self._render_quotes(us_stocks),
            "----------------",
            "🇹🇼 **台股**",
            self._render_quotes(tw_stocks),
            "----------------",
            self._render_report(report),
        ]
        if notion_url:
            sections.append(f"📒 [**在 Notion 查看完整深度分析報告**]({notion_url})")
        return self._build_payload("投資情報報告", "\n".join(part for part in sections if part))

    def _build_ai_payload(self, report: AnalyzedReport, notion_url: str | None) -> dict:
        sections = [self._render_report(report)]
        if notion_url:
            sections.append(f"📒 [**在 Notion 查看完整 AI 技術情報**]({notion_url})")
        return self._build_payload("AI 技術前沿情報", "\n".join(part for part in sections if part))

    def _render_quotes(self, quotes: list[dict]) -> str:
        if not quotes:
            return "本輪未取得資料"
        return "\n\n".join(
            f"**{quote['symbol']}**\n現:{self._format_quote_value(quote, 'price')} | 變:{self._format_quote_change(quote)}\n區:{self._format_quote_range(quote)}"
            for quote in quotes
        )

    def _format_quote_value(self, quote: dict, key: str) -> str:
        value = quote.get(key)
        if value is None:
            return "-"
        numeric = float(value)
        if self._is_tw_quote(quote):
            return self._format_tw_number(numeric)
        return f"{numeric:.2f}"

    def _format_quote_change(self, quote: dict) -> str:
        change = quote.get("change", 0)
        if isinstance(change, str):
            try:
                change_value = float(change)
            except ValueError:
                return change
        else:
            change_value = float(change)
        if self._is_tw_quote(quote):
            return self._format_tw_signed_number(change_value)
        return f"{change_value:+.2f}"

    def _format_quote_range(self, quote: dict) -> str:
        range_value = quote.get("range")
        if not range_value:
            return "-"
        if isinstance(range_value, str) and "-" in range_value:
            low_raw, high_raw = range_value.split("-", 1)
            low = float(low_raw)
            high = float(high_raw)
            if self._is_tw_quote(quote):
                return f"{self._format_tw_number(low)}-{self._format_tw_number(high)}"
            return f"{low:.2f}-{high:.2f}"
        return str(range_value)

    def _is_tw_quote(self, quote: dict) -> bool:
        symbol = str(quote.get("symbol", ""))
        return symbol.isdigit()

    def _format_tw_number(self, value: float) -> str:
        rendered = f"{value:.2f}"
        if rendered.endswith(".00"):
            return rendered[:-3]
        return rendered

    def _format_tw_signed_number(self, value: float) -> str:
        sign = "+" if value >= 0 else "-"
        rendered = self._format_tw_number(abs(value))
        return f"{sign}{rendered}"

    def _render_report(self, report: AnalyzedReport) -> str:
        lines: list[str] = []
        if report.summary:
            lines.append(f"📌 **摘要**\n{self._truncate_text(report.summary, 220)}")

        display_items = self._dedupe_report_items(report.items)
        for item in display_items[: self.DISCORD_ITEM_LIMIT]:
            lines.append("----------------")
            lines.append(f"**[{item.title}]({item.url})**")
            if item.summary:
                lines.append(f"• {self._truncate_text(item.summary, self.SUMMARY_LIMIT)}")
            if item.insight:
                lines.append(f"> 💡 {self._truncate_text(item.insight, self.INSIGHT_LIMIT)}")

        if report.outlook:
            # Keep the final conclusion substantially complete; prioritize truncating earlier snippets instead.
            lines.append(f"{report.outlook_label}\n{self._truncate_text(report.outlook, self.OUTLOOK_LIMIT)}")
        remaining = max(len(display_items) - self.DISCORD_ITEM_LIMIT, 0)
        if remaining:
            lines.append("")
            lines.append(f"📎 其餘 {remaining} 則延伸內容與來源細節請看 Notion。")
        return "\n".join(lines).strip()

    def _dedupe_report_items(self, items: list) -> list:
        deduped: list = []
        seen_keys: set[str] = set()
        for item in items:
            key = content_dedupe_key(
                title=getattr(item, "title", ""),
                url=getattr(item, "url", ""),
                source_name=getattr(item, "source_name", ""),
                summary=getattr(item, "summary", ""),
            )
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
        return deduped

    def _build_payload(self, title: str, description: str) -> dict:
        return {
            "embeds": [
                {
                    "title": title,
                    "description": description[:4096],
                    "color": 0x3498DB if "AI" in title else 0x2ECC71,
                }
            ]
        }

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _deliver(self, payload: dict) -> None:
        if self.dry_run or not self.enabled:
            logger.info("Discord delivery skipped (dry_run=%s, enabled=%s).", self.dry_run, self.enabled)
            return
        if not Config.DISCORD_WEBHOOK_URL:
            logger.warning("Discord webhook missing, skipping send.")
            return

        response = self.session.post(
            Config.DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        logger.info("Sent report to Discord.")
