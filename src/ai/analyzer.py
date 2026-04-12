from __future__ import annotations

import json
import signal
from collections import Counter
from dataclasses import asdict
from datetime import timezone
from typing import Callable

from src.config import Config
from src.models import AnalyzedReport, IntelligenceItem, ReportItem
from src.pipeline import canonicalize_url, parse_published_at, summarize_sources
from src.utils.logger import logger

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency in dry-run
    genai = None

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional dependency in dry-run
    Groq = None


class TimeoutException(Exception):
    """Raised when upstream AI calls exceed the hard timeout."""


def timeout_handler(signum, frame):  # pragma: no cover - signal plumbing
    raise TimeoutException


class AIAnalyzer:
    _REQUIRED_SOURCE_TYPES = ("official_news", "model_release", "github_release")
    _SOURCE_RANK = {
        "official_news": 0,
        "github_release": 1,
        "model_release": 2,
        "research": 3,
        "github_repo": 4,
        "news": 5,
        "community": 6,
        "unknown": 9,
    }

    def __init__(self, enable_ai: bool | None = None):
        self.enable_ai = Config.ENABLE_AI_ANALYSIS if enable_ai is None else enable_ai
        self.gemini_client = (
            genai.Client(api_key=Config.GEMINI_API_KEY)
            if self.enable_ai and genai and Config.GEMINI_API_KEY
            else None
        )
        self.groq_client = (
            Groq(api_key=Config.GROQ_API_KEY)
            if self.enable_ai and Groq and Config.GROQ_API_KEY
            else None
        )
        if self.enable_ai:
            logger.info("Analyzer initialized with live AI providers.")
        else:
            logger.info("Analyzer running in local synthesis mode.")

    def analyze_stock_market(
        self,
        stock_quotes: list[dict],
        news: list[IntelligenceItem],
    ) -> AnalyzedReport:
        fallback = lambda: self._build_stock_fallback(stock_quotes, news)
        if not news:
            return fallback()

        prompt = {
            "task": "analyze_stock_market",
            "language": "zh-TW",
            "stocks": stock_quotes,
            "news": [asdict(item) for item in news[:12]],
            "instructions": [
                "回傳 JSON，禁止輸出 markdown 或額外說明。",
                "summary 要總結盤勢、新聞主線與風險。",
                "items 最多 5 筆，每筆要有 title,url,summary,insight。",
                "outlook 要給後續觀察重點，避免過度肯定的投資建議。",
            ],
        }
        return self._run_analysis(
            prompt=prompt,
            fallback=fallback,
            title="投資情報報告",
            outlook_label="🕵️ 專家總結",
        )

    def analyze_ai_tech(self, news: list[IntelligenceItem]) -> AnalyzedReport:
        fallback = lambda: self._build_ai_fallback(news)
        if not news:
            return fallback()

        prompt = {
            "task": "analyze_ai_tech",
            "language": "zh-TW",
            "news": [asdict(item) for item in news[:15]],
            "instructions": [
                "回傳 JSON，禁止輸出 markdown 或額外說明。",
                "summary 要總結今天值得看的模型、agent、SDK、GitHub 專案、論文與官方消息。",
                "items 最多 6 筆，每筆要有 title,url,summary,insight。",
                "優先挑選官方發布、GitHub release、快速成長的專案、研究論文，不要挑法律糾紛或社會新聞。",
                "若有版本號、模型名稱、產品名，必須完整保留。",
                "outlook 要指出接下來值得追蹤的官方來源或技術方向。",
            ],
        }
        report = self._run_analysis(
            prompt=prompt,
            fallback=fallback,
            title="AI 技術前沿情報",
            outlook_label="🔮 未來展望",
        )
        return self._post_process_ai_report(report, news)

    def _run_analysis(
        self,
        *,
        prompt: dict,
        fallback: Callable[[], AnalyzedReport],
        title: str,
        outlook_label: str,
    ) -> AnalyzedReport:
        if not self.enable_ai:
            return fallback()

        raw_response = self._get_ai_response(json.dumps(prompt, ensure_ascii=False))
        if not raw_response:
            return fallback()

        parsed = self._parse_response(raw_response, title=title, outlook_label=outlook_label)
        if parsed is None:
            logger.warning("AI response was not valid JSON. Falling back to local synthesis.")
            return fallback()
        return parsed

    def _parse_response(
        self,
        raw_response: str,
        *,
        title: str,
        outlook_label: str,
    ) -> AnalyzedReport | None:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        items: list[ReportItem] = []
        for item in payload.get("items", [])[:6]:
            if not isinstance(item, dict):
                continue
            title_text = self._safe_text(item.get("title"))
            url_text = self._safe_text(item.get("url"))
            if not title_text or not url_text:
                continue
            items.append(
                ReportItem(
                    title=title_text,
                    url=url_text,
                    summary=self._safe_text(item.get("summary")),
                    insight=self._safe_text(item.get("insight")),
                    source_name=self._safe_text(item.get("source_name"), default="unknown") or "unknown",
                    source_type=self._safe_text(item.get("source_type"), default="unknown") or "unknown",
                    published_at=self._safe_text(item.get("published_at")) or None,
                )
            )

        if not items:
            return None

        return AnalyzedReport(
            title=title,
            summary=self._safe_text(payload.get("summary")),
            items=items,
            outlook=self._safe_text(payload.get("outlook")),
            outlook_label=outlook_label,
        )

    def _build_stock_fallback(
        self,
        stock_quotes: list[dict],
        news: list[IntelligenceItem],
    ) -> AnalyzedReport:
        moves = [f"{quote['symbol']} {quote['price']} ({quote['change']})" for quote in stock_quotes]
        source_summary = summarize_sources(news)
        items = [
            ReportItem(
                title=item.title,
                url=item.url,
                summary=item.desc or "原始來源未提供摘要。",
                insight=(
                    f"來源屬於 {item.source_name} / {item.source_type}。"
                    " 目前是離線合成模式，建議將此則與最新財報、法說會或價格走勢交叉驗證。"
                ),
                source_name=item.source_name,
                source_type=item.source_type,
                published_at=item.published_at,
            )
            for item in news[:5]
        ]
        return AnalyzedReport(
            title="投資情報報告",
            summary=(
                f"本輪共整理 {len(news)} 則市場相關消息，來源分布為 {source_summary}。"
                f" 關注標的即時報價包含：{'; '.join(moves) if moves else '本輪未取得報價'}。"
            ),
            items=items,
            outlook=(
                "先確認價格來源是否處於交易時段，再針對高關聯新聞做基本面與事件驅動拆解，"
                "避免把延遲報價和舊聞直接餵進投資判斷。"
            ),
            outlook_label="🕵️ 專家總結",
            metadata={"mode": "fallback"},
        )

    def _build_ai_fallback(self, news: list[IntelligenceItem]) -> AnalyzedReport:
        source_counter = Counter(item.source_name for item in news if item.source_name)
        hot_sources = "、".join(name for name, _ in source_counter.most_common(4)) or "無"
        items = [
            ReportItem(
                title=item.title,
                url=item.url,
                summary=item.desc or "原始來源未提供摘要。",
                insight=(
                    f"此則來自 {item.source_name}，分類為 {item.source_type}。"
                    " 建議把它放回官方公告、GitHub release 或論文頁面做二次查證。"
                ),
                source_name=item.source_name,
                source_type=item.source_type,
                published_at=item.published_at,
            )
            for item in news[:6]
        ]
        return AnalyzedReport(
            title="AI 技術前沿情報",
            summary=(
                f"本輪共整理 {len(news)} 則 AI 情報，主要來源涵蓋 {hot_sources}。"
                " 目前為本地合成模式，重點在保留來源脈絡與後續可追蹤性。"
            ),
            items=items,
            outlook=(
                "接下來應優先追官方發布、GitHub release、arXiv 與 Hugging Face 指標變化，"
                "再把社群熱度當次要訊號，而不是主排序依據。"
            ),
            outlook_label="🔮 未來展望",
            metadata={"mode": "fallback"},
        )

    def _post_process_ai_report(self, report: AnalyzedReport, news: list[IntelligenceItem]) -> AnalyzedReport:
        if not report.items:
            return report
        self._hydrate_ai_item_metadata(report, news)
        self._enforce_ai_signal_coverage(report, news)
        return report

    def _hydrate_ai_item_metadata(self, report: AnalyzedReport, news: list[IntelligenceItem]) -> None:
        by_url: dict[str, IntelligenceItem] = {}
        by_title_key: dict[str, IntelligenceItem] = {}
        for item in news:
            item_url = canonicalize_url(item.url)
            if item_url:
                by_url[item_url] = item
            title_key = self._title_key(item.title)
            if title_key and title_key not in by_title_key:
                by_title_key[title_key] = item

        for report_item in report.items:
            matched = by_url.get(canonicalize_url(report_item.url)) or by_title_key.get(self._title_key(report_item.title))
            if not matched:
                continue
            if not report_item.source_name or report_item.source_name == "unknown":
                report_item.source_name = matched.source_name
            if not report_item.source_type or report_item.source_type == "unknown":
                report_item.source_type = matched.source_type
            if not report_item.published_at:
                report_item.published_at = matched.published_at

    def _enforce_ai_signal_coverage(self, report: AnalyzedReport, news: list[IntelligenceItem]) -> None:
        must_include: list[IntelligenceItem] = []

        for source_type in self._REQUIRED_SOURCE_TYPES:
            if any(item.source_type == source_type for item in report.items):
                continue
            candidate = self._find_source_candidate(news, source_type)
            if candidate:
                must_include.append(candidate)

        # Preserve at least two newest items from the latest official source cluster.
        official_candidates = [item for item in news if item.source_type == "official_news"]
        if official_candidates:
            official_candidates.sort(key=lambda item: -self._published_ts(item.published_at))
            latest_source_name = official_candidates[0].source_name
            latest_cluster = [item for item in official_candidates if item.source_name == latest_source_name]
            latest_cluster.sort(key=lambda item: -self._published_ts(item.published_at))
            for candidate in latest_cluster[:2]:
                if not self._report_has_item(report.items, candidate):
                    must_include.append(candidate)

        for candidate in reversed(must_include):
            if self._report_has_item(report.items, candidate):
                continue
            report.items.insert(0, self._report_item_from_news(candidate))

        deduped: list[ReportItem] = []
        seen_keys: set[str] = set()
        for item in report.items:
            key = canonicalize_url(item.url) or self._title_key(item.title)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)

        deduped.sort(
            key=lambda item: (
                self._SOURCE_RANK.get(item.source_type, self._SOURCE_RANK["unknown"]),
                -self._published_ts(item.published_at),
            )
        )
        report.items = deduped[:6]

    def _find_source_candidate(self, news: list[IntelligenceItem], source_type: str) -> IntelligenceItem | None:
        candidates = [item for item in news if item.source_type == source_type]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                self._SOURCE_RANK.get(item.source_type, self._SOURCE_RANK["unknown"]),
                -self._published_ts(item.published_at),
            )
        )
        return candidates[0]

    def _report_item_from_news(self, item: IntelligenceItem) -> ReportItem:
        return ReportItem(
            title=item.title,
            url=item.url,
            summary=item.desc or "原始來源未提供摘要。",
            insight=f"此則來自 {item.source_name}，屬於 {item.source_type}，為近期高優先議題。",
            source_name=item.source_name,
            source_type=item.source_type,
            published_at=item.published_at,
        )

    def _report_has_item(self, report_items: list[ReportItem], candidate: IntelligenceItem) -> bool:
        candidate_url = canonicalize_url(candidate.url)
        candidate_title = self._title_key(candidate.title)
        for report_item in report_items:
            report_url = canonicalize_url(report_item.url)
            report_title = self._title_key(report_item.title)
            if candidate_url and report_url and candidate_url == report_url:
                return True
            if candidate_title and report_title and candidate_title == report_title:
                return True
        return False

    def _published_ts(self, value: str | None) -> float:
        parsed = parse_published_at(value)
        if parsed is None:
            return float("-inf")
        return parsed.astimezone(timezone.utc).timestamp()

    def _title_key(self, title: str | None) -> str:
        return "".join(ch for ch in (title or "").lower() if ch.isalnum())[:120]

    def _safe_text(self, value, *, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            return json.dumps(value, ensure_ascii=False).strip()
        if isinstance(value, list):
            parts = [self._safe_text(part) for part in value]
            joined = " ".join(part for part in parts if part)
            return joined.strip()
        return str(value).strip()

    def _get_ai_response(self, prompt: str) -> str | None:
        if self.gemini_client:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                response = self.gemini_client.models.generate_content(
                    model=Config.AI_MODEL,
                    contents=prompt,
                )
                signal.alarm(0)
                return response.text
            except Exception as exc:  # pragma: no cover - live provider failure
                signal.alarm(0)
                logger.warning("Gemini failed, trying Groq fallback: %s", exc)

        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                return response.choices[0].message.content
            except Exception as exc:  # pragma: no cover - live provider failure
                logger.error("Groq fallback failed: %s", exc)

        return None
