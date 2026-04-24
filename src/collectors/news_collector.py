from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from src.config import Config
from src.utils.logger import logger


class NewsCollector:
    _NEWSAPI_Q_MAX_CHARS = 500
    _NEWSAPI_Q_BUDGET_CHARS = 450

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
        domains = "openai.com,platform.openai.com,anthropic.com,code.claude.com,github.blog,blog.google,ai.google.dev,x.ai,docs.x.ai,huggingface.co,techcrunch.com,venturebeat.com,theverge.com,arstechnica.com,cbsnews.com,theguardian.com,fortune.com,pcworld.com,tomshardware.com"
        keywords = [
            "Claude Mythos",
            "Project Glasswing",
            "Mythos Preview",
            "AI zero-day",
            "AI cybersecurity",
            "AI vulnerability",
            "unauthorized access AI model",
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
            "Claude Code",
            "Ultraplan",
            "Google Gemini",
            "NotebookLM",
            "notebooks",
            "Google Gemma",
            "Gemma",
            "xAI",
            "Grok",
            "Llama",
            "Qwen",
            "Mistral",
            "DeepSeek",
            "Hugging Face",
            "GitHub Copilot",
        ]
        broad = self._fetch_keyword_batches(
            keywords,
            domains=domains,
            page_size=20,
            days_back=Config.AI_NEWS_LOOKBACK_DAYS,
            source_type="news",
        )
        feature_focus = self._fetch_keyword_batches(
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
                "Claude Code",
                "Ultraplan",
                "NotebookLM",
                "notebooks",
                "Grok multi-agent",
                "Gemini Interactions API",
                "Claude Mythos breach",
                "Project Glasswing",
                "zero-day vulnerabilities",
                "frontier model cybersecurity",
            ],
            domains="anthropic.com,code.claude.com,openai.com,platform.openai.com,github.blog,ai.google.dev,x.ai,docs.x.ai,techcrunch.com,venturebeat.com,theverge.com,arstechnica.com,cbsnews.com,theguardian.com,fortune.com,pcworld.com,tomshardware.com",
            page_size=10,
            days_back=Config.AI_NEWS_LOOKBACK_DAYS,
            source_type="news",
        )
        return broad + feature_focus

    def _fetch_keyword_batches(
        self,
        keywords: list[str],
        *,
        domains: str | None,
        page_size: int,
        days_back: int,
        source_type: str,
    ) -> list[dict]:
        results: list[dict] = []
        for batch in self._split_keyword_batches(keywords):
            results.extend(
                self._fetch_by_keywords(
                    batch,
                    domains=domains,
                    page_size=page_size,
                    days_back=days_back,
                    source_type=source_type,
                )
            )
        return results

    def _split_keyword_batches(self, keywords: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        for keyword in keywords:
            candidate = [*current, keyword]
            if current and len(self._build_query(candidate)) > self._NEWSAPI_Q_BUDGET_CHARS:
                batches.append(current)
                current = [keyword]
            else:
                current = candidate
        if current:
            batches.append(current)
        return batches

    def _build_query(self, keywords: list[str]) -> str:
        return "(" + " OR ".join(f'"{keyword}"' for keyword in keywords) + ")"

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
        query = self._build_query(keywords)
        if len(query) > self._NEWSAPI_Q_MAX_CHARS:
            logger.error("News API query exceeds %s chars; split it before fetching.", self._NEWSAPI_Q_MAX_CHARS)
            return []
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
