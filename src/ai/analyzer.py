from __future__ import annotations

import json
import signal
from collections import Counter
from dataclasses import asdict
from typing import Callable

from src.config import Config
from src.models import AnalyzedReport, IntelligenceItem, ReportItem
from src.pipeline import summarize_sources
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
        return self._run_analysis(
            prompt=prompt,
            fallback=fallback,
            title="AI 技術前沿情報",
            outlook_label="🔮 未來展望",
        )

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

        items: list[ReportItem] = []
        for item in payload.get("items", [])[:6]:
            if not item.get("title") or not item.get("url"):
                continue
            items.append(
                ReportItem(
                    title=item["title"].strip(),
                    url=item["url"].strip(),
                    summary=item.get("summary", "").strip(),
                    insight=item.get("insight", "").strip(),
                    source_name=item.get("source_name", "unknown").strip() or "unknown",
                    source_type=item.get("source_type", "unknown").strip() or "unknown",
                    published_at=item.get("published_at"),
                )
            )

        if not items:
            return None

        return AnalyzedReport(
            title=title,
            summary=payload.get("summary", "").strip(),
            items=items,
            outlook=payload.get("outlook", "").strip(),
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
