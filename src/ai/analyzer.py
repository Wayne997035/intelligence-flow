from __future__ import annotations

import json
import re
import signal
from collections import Counter
from dataclasses import asdict
from datetime import timezone
from typing import Callable

from src.config import Config
from src.models import AnalyzedReport, IntelligenceItem, ReportItem
from src.pipeline import canonicalize_url, normalize_text, parse_published_at, summarize_sources
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
    _AI_REPORT_ITEM_LIMIT = 10
    _REQUIRED_SOURCE_TYPES = ("official_news", "model_release", "github_release")
    _CORE_PROVIDERS = ("anthropic", "openai", "google", "xai")
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
            "news": [asdict(item) for item in news[:24]],
            "instructions": [
                "回傳 JSON，禁止輸出 markdown 或額外說明。",
                "summary 要總結今天值得看的模型、agent、SDK、GitHub 專案、論文與官方消息。",
                "items 最多 10 筆，每筆要有 title,url,summary,insight。",
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

    def build_ai_brief_item(self, item: IntelligenceItem | ReportItem | dict) -> dict:
        if isinstance(item, ReportItem):
            title = normalize_text(item.title)
            url = normalize_text(item.url)
            desc = normalize_text(item.summary)
            source_name = normalize_text(item.source_name) or "unknown"
            source_type = normalize_text(item.source_type) or "unknown"
            published_at = normalize_text(item.published_at) or None
        elif isinstance(item, IntelligenceItem):
            title = normalize_text(item.title)
            url = normalize_text(item.url)
            desc = normalize_text(item.desc)
            source_name = normalize_text(item.source_name) or "unknown"
            source_type = normalize_text(item.source_type) or "unknown"
            published_at = normalize_text(item.published_at) or None
        else:
            title = normalize_text(item.get("title"))
            url = normalize_text(item.get("url"))
            desc = normalize_text(item.get("desc") or item.get("summary"))
            source_name = normalize_text(item.get("source_name")) or "unknown"
            source_type = normalize_text(item.get("source_type")) or "unknown"
            published_at = normalize_text(item.get("published_at")) or None

        headline = self._normalize_brief_title(title)
        summary = self._summarize_brief_item(headline, desc, source_type)
        insight = self._summarize_brief_insight(headline, source_name, source_type)
        return {
            "title": title,
            "url": url,
            "summary": summary,
            "insight": insight,
            "source_name": source_name,
            "source_type": source_type,
            "published_at": published_at,
        }

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
        for item in payload.get("items", [])[: self._AI_REPORT_ITEM_LIMIT]:
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
        items: list[ReportItem] = []
        for item in news[: self._AI_REPORT_ITEM_LIMIT]:
            brief = self.build_ai_brief_item(item)
            items.append(
                ReportItem(
                    title=item.title,
                    url=item.url,
                    summary=brief["summary"] or item.desc or "原始來源未提供摘要。",
                    insight=brief["insight"],
                    source_name=item.source_name,
                    source_type=item.source_type,
                    published_at=item.published_at,
                )
            )
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
        self._fill_missing_ai_item_insights(report, news)
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

        # Preserve recent official coverage from distinct sources instead of letting
        # one publisher dominate the whole report.
        official_candidates = [item for item in news if item.source_type == "official_news"]
        if official_candidates:
            official_candidates.sort(key=lambda item: -self._published_ts(item.published_at))
            preferred_officials: list[IntelligenceItem] = []
            seen_sources: set[str] = set()
            for candidate in official_candidates:
                source_name = (candidate.source_name or "").strip().lower()
                if not source_name or source_name in seen_sources:
                    continue
                preferred_officials.append(candidate)
                seen_sources.add(source_name)
                if len(preferred_officials) >= 2:
                    break
            for candidate in preferred_officials:
                if not self._report_has_item(report.items, candidate):
                    must_include.append(candidate)

        for provider in self._CORE_PROVIDERS:
            if any(self._provider_key_from_report(item) == provider for item in report.items):
                continue
            candidate = self._find_provider_candidate(news, provider)
            if candidate and not self._report_has_item(report.items, candidate):
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
        report.items = deduped[: self._AI_REPORT_ITEM_LIMIT]

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
        brief = self.build_ai_brief_item(item)
        return ReportItem(
            title=item.title,
            url=item.url,
            summary=brief["summary"] or item.desc or "原始來源未提供摘要。",
            insight=brief["insight"],
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

    def _find_provider_candidate(self, news: list[IntelligenceItem], provider: str) -> IntelligenceItem | None:
        candidates = [item for item in news if self._provider_key_from_news(item) == provider]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                self._SOURCE_RANK.get(item.source_type, self._SOURCE_RANK["unknown"]),
                -self._published_ts(item.published_at),
            )
        )
        return candidates[0]

    def _provider_key_from_news(self, item: IntelligenceItem) -> str | None:
        return self._provider_key(item.source_name, item.title, item.url)

    def _provider_key_from_report(self, item: ReportItem) -> str | None:
        return self._provider_key(item.source_name, item.title, item.url)

    def _provider_key(self, source_name: str | None, title: str | None, url: str | None) -> str | None:
        haystack = f"{source_name or ''} {title or ''} {url or ''}".lower()
        if any(keyword in haystack for keyword in ("anthropic", "claude", "code.claude.com")):
            return "anthropic"
        if any(keyword in haystack for keyword in ("openai", "chatgpt", "gpt", "platform.openai.com")):
            return "openai"
        if any(keyword in haystack for keyword in ("google", "gemini", "gemma", "ai.google.dev")):
            return "google"
        if any(keyword in haystack for keyword in ("xai", "grok", "docs.x.ai", "x.ai")):
            return "xai"
        return None

    def _published_ts(self, value: str | None) -> float:
        parsed = parse_published_at(value)
        if parsed is None:
            return float("-inf")
        return parsed.astimezone(timezone.utc).timestamp()

    def _title_key(self, title: str | None) -> str:
        return "".join(ch for ch in (title or "").lower() if ch.isalnum())[:120]

    def _fill_missing_ai_item_insights(self, report: AnalyzedReport, news: list[IntelligenceItem]) -> None:
        by_url: dict[str, IntelligenceItem] = {}
        by_title_key: dict[str, IntelligenceItem] = {}
        for item in news:
            if item.url:
                by_url[canonicalize_url(item.url)] = item
            title_key = self._title_key(item.title)
            if title_key and title_key not in by_title_key:
                by_title_key[title_key] = item

        for report_item in report.items:
            if report_item.insight:
                continue
            matched = by_url.get(canonicalize_url(report_item.url)) or by_title_key.get(self._title_key(report_item.title))
            brief = self.build_ai_brief_item(matched or report_item)
            report_item.insight = brief["insight"]

    def _normalize_brief_title(self, title: str) -> str:
        normalized = normalize_text(title)
        normalized = re.sub(r"^\[(Official|HF Model|HF Paper|GitHub Release|GitHub)\]\s*", "", normalized, flags=re.IGNORECASE)
        if len(normalized) > 110:
            normalized = normalized[:107].rstrip() + "..."
        return normalized

    def _summarize_brief_item(self, headline: str, desc: str, source_type: str) -> str:
        details = normalize_text(desc)
        if source_type == "github_repo":
            return f"GitHub 熱門專案，聚焦 {headline}。"
        if source_type == "github_release":
            return f"GitHub Release 更新重點：{headline}。"
        if source_type == "model_release":
            task_match = re.search(r"Task:\s*([^|]+)", details, flags=re.IGNORECASE)
            downloads_match = re.search(r"Downloads:\s*([0-9,]+)", details, flags=re.IGNORECASE)
            extras = []
            if task_match:
                extras.append(f"任務類型為 {task_match.group(1).strip()}")
            if downloads_match:
                extras.append(f"下載量約 {downloads_match.group(1).strip()}")
            suffix = f"，{'，'.join(extras)}" if extras else ""
            return f"模型更新重點：{headline}{suffix}。"
        if source_type == "community":
            return f"社群近期討論焦點：{headline}。"
        if source_type == "research":
            return f"研究論文重點：{headline}。"
        if details and "collected from" not in details.lower():
            sentence = details[:160].rstrip()
            if not sentence.endswith(("。", ".", "!", "?")):
                sentence += "。"
            return sentence
        return f"近期值得追蹤的更新：{headline}。"

    def _summarize_brief_insight(self, headline: str, source_name: str, source_type: str) -> str:
        headline_lower = headline.lower()
        provider = self._brief_provider_label(source_name, headline)

        if "advisor" in headline_lower:
            return (
                f"{provider} 正在把規劃與執行拆成雙模型協作，"
                "這代表長流程 Agent workflow 開始走向分層控制。"
            )
        if "managed agent" in headline_lower:
            return (
                f"{provider} 把 agent runtime 官方化，"
                "代表競爭重點已從單純模型能力延伸到可直接落地的執行基礎設施。"
            )
        if "ant cli" in headline_lower or "claude code" in headline_lower:
            return (
                f"{provider} 正把 API、CLI 與工程流程串起來，"
                "這會加速 AI 能力往終端、自動化腳本與本地開發環境滲透。"
            )
        if "notebook" in headline_lower or "notebooklm" in headline_lower:
            return (
                f"{provider} 正把聊天、檔案與研究流程收斂成專案容器，"
                "這有助於提升長任務的上下文延續性與知識管理效率。"
            )
        if "mythos" in headline_lower or "cyber" in headline_lower or "glasswing" in headline_lower:
            return (
                f"{provider} 開始把模型能力往高風險垂直場景做產品分層，"
                "顯示安全與專業領域模型會成為下一階段差異化重點。"
            )
        if ("chatgpt" in headline_lower and "$100" in headline_lower) or "100 per month" in headline_lower:
            return (
                "OpenAI 正把高強度 ChatGPT / Codex 工作負載往更高價位方案分層，"
                "反映專業用戶的生產力需求已足以支撐更進一步的商業化。"
            )
        if "3d model" in headline_lower or "simulation" in headline_lower:
            return (
                f"{provider} 正把回覆形式從純文字擴展到可操作的視覺化內容，"
                "產品定位會更接近能直接幫使用者完成理解與演示的工作介面。"
            )

        source_label = source_name or source_type or "來源"
        if source_type == "official_news":
            return f"{source_label} 的這則更新屬於主線產品動態，適合放進本輪重點追蹤。"
        if source_type == "news":
            return f"這則媒體報導可作為 {provider} 官方動向的外部驗證，適合搭配原始公告一起判讀。"
        if source_type == "model_release":
            return f"{headline} 代表近期模型供給的變化，值得和其他主流模型能力、部署成本與任務定位一起比較。"
        if source_type == "research":
            return f"這篇研究可視為後續技術方向訊號，適合和產品更新一起觀察，判斷哪些能力可能先進入產品化。"
        if source_type == "community":
            return f"這則社群討論能補足 {provider} 官方公告之外的實作體感，但仍要和一手來源交叉驗證。"
        if source_type in {"github_repo", "github_release"}:
            return f"這則開源動態可補充 {provider} 生態系進展，適合作為主線產品之外的延伸觀察。"
        return f"這則內容來自 {source_label}，建議結合原文一起判斷它在本輪資訊中的重要性。"

    def _brief_provider_label(self, source_name: str, headline: str) -> str:
        haystack = f"{source_name} {headline}".lower()
        if any(keyword in haystack for keyword in ("claude", "anthropic")):
            return "Anthropic"
        if any(keyword in haystack for keyword in ("chatgpt", "openai", "gpt", "codex")):
            return "OpenAI"
        if any(keyword in haystack for keyword in ("gemini", "google", "gemma", "notebooklm")):
            return "Google"
        if any(keyword in haystack for keyword in ("grok", "xai", "x.ai")):
            return "xAI"
        return source_name or "這個來源"

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
                    model=Config.GROQ_MODEL,
                )
                return response.choices[0].message.content
            except Exception as exc:  # pragma: no cover - live provider failure
                logger.error("Groq fallback failed: %s", exc)

        return None
