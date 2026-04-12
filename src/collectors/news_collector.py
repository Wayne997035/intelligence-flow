from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from src.config import Config
from src.utils.logger import logger


class NewsCollector:
    def __init__(self):
        self.api_key = Config.NEWS_API_KEY

    def fetch_stock_news(self) -> list[dict]:
        domains = "reuters.com,bloomberg.com,wsj.com,cnbc.com,techcrunch.com,finance.yahoo.com"
        stock_kws = Config.US_STOCKS + Config.TW_STOCKS + [
            "Nvidia Blackwell",
            "TSMC 2nm",
            "AI chip demand",
            "Semiconductor supply chain",
            "Earnings report",
            "Price target",
        ]
        return self._fetch_by_keywords(
            stock_kws,
            domains=domains,
            page_size=10,
            days_back=Config.STOCK_NEWS_LOOKBACK_DAYS,
            source_type="news",
        )

    def fetch_ai_tech_news(self) -> list[dict]:
        domains = "openai.com,anthropic.com,github.blog,blog.google,ai.google.dev,huggingface.co,techcrunch.com,venturebeat.com,theverge.com,arstechnica.com"
        keywords = [
            "model release",
            "reasoning model",
            "AI agent",
            "agentic workflow",
            "AI skill",
            "managed agents",
            "Claude managed agents",
            "Anthropic managed agents",
            "OpenAI",
            "ChatGPT",
            "Codex",
            "Anthropic",
            "Google Gemini",
            "xAI",
            "Grok",
            "Llama",
            "Qwen",
            "Mistral",
            "DeepSeek",
            "Hugging Face",
            "GitHub Copilot",
        ]
        broad = self._fetch_by_keywords(
            keywords,
            domains=domains,
            page_size=20,
            days_back=Config.AI_NEWS_LOOKBACK_DAYS,
            source_type="news",
        )
        feature_focus = self._fetch_by_keywords(
            [
                "managed agents",
                "agent framework",
                "agent skill",
                "agent tooling",
                "tool use",
                "workflow automation",
                "Codex",
                "Grok",
                "ChatGPT",
            ],
            domains="anthropic.com,openai.com,github.blog,techcrunch.com,venturebeat.com,theverge.com,arstechnica.com",
            page_size=10,
            days_back=Config.AI_NEWS_LOOKBACK_DAYS,
            source_type="news",
        )
        return broad + feature_focus

    def _fetch_by_keywords(
        self,
        keywords: list[str],
        *,
        domains: str | None,
        page_size: int,
        days_back: int,
        source_type: str,
    ) -> list[dict]:
        if not self.api_key:
            logger.warning("News API key missing.")
            return []

        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = "(" + " OR ".join(f'"{keyword}"' for keyword in keywords) + ")"
        params = {
            "apiKey": self.api_key,
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
        }
        if domains:
            params["domains"] = domains

        try:
            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - live source failures
            logger.error("Error fetching news: %s", exc)
            return []

        articles = payload.get("articles", [])
        return [
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "desc": article.get("description") or "",
                "source_name": article.get("source", {}).get("name", "newsapi"),
                "source_type": source_type,
                "published_at": article.get("publishedAt"),
            }
            for article in articles
            if article.get("title") and article.get("url")
        ]
