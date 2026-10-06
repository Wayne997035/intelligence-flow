from __future__ import annotations

from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from src.config import Config
from src.utils.logger import logger

try:
    import feedparser
except ImportError:  # pragma: no cover - optional in minimal env
    feedparser = None


class GoogleNewsCollector:
    """Keyless topic search via Google News RSS.

    NewsAPI's free tier is limited to a fixed domain list and 100 req/day, so
    narrow topics (e.g. SpaceX launches, a specific model release) often come
    back empty. Google News RSS aggregates far more publishers and needs no key,
    which makes it a good completeness backstop for both reports.
    """

    _BASE_URL = "https://news.google.com/rss/search"

    def fetch_stock_topics(self, *, limit_per_topic: int = 5) -> list[dict]:
        return self._fetch_topics(
            Config.STOCK_WATCH_TOPICS,
            days_back=Config.STOCK_NEWS_LOOKBACK_DAYS,
            limit_per_topic=limit_per_topic,
        )

    def fetch_ai_topics(self, *, limit_per_topic: int = 5) -> list[dict]:
        return self._fetch_topics(
            Config.AI_WATCH_TOPICS,
            days_back=Config.AI_NEWS_LOOKBACK_DAYS,
            limit_per_topic=limit_per_topic,
        )

    def build_url(self, topic: str, *, days_back: int) -> str:
        query = quote_plus(f"{topic} when:{max(days_back, 1)}d")
        return f"{self._BASE_URL}?q={query}&hl=en-US&gl=US&ceid=US:en"

    def _fetch_topics(self, topics: list[str], *, days_back: int, limit_per_topic: int) -> list[dict]:
        if not Config.ENABLE_GOOGLE_NEWS:
            return []
        if feedparser is None:
            logger.warning("feedparser not installed; skipping Google News topics.")
            return []

        results: list[dict] = []
        for topic in topics:
            try:
                feed = feedparser.parse(self.build_url(topic, days_back=days_back))
            except Exception as exc:  # pragma: no cover - feed parser edge cases
                logger.warning("Google News fetch failed for %s: %s", topic, exc)
                continue
            picked = 0
            for entry in getattr(feed, "entries", []):
                item = self._normalize_entry(entry, topic)
                if item is None:
                    continue
                results.append(item)
                picked += 1
                if picked >= limit_per_topic:
                    break
        return results

    def _normalize_entry(self, entry, topic: str) -> dict | None:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            return None

        source = entry.get("source") or {}
        source_name = (source.get("title") if hasattr(source, "get") else "") or "Google News"
        # Google News appends " - Publisher" to every headline; strip it so
        # cross-source title dedup can match the same story from NewsAPI.
        suffix = f" - {source_name}"
        if title.endswith(suffix):
            title = title[: -len(suffix)].rstrip()

        return {
            "title": title,
            "url": link,
            "desc": "",
            "tags": [topic],
            "source_name": source_name,
            "source_type": "news",
            "published_at": self._published_at(entry),
        }

    def _published_at(self, entry) -> str | None:
        raw = entry.get("published") or entry.get("updated")
        if not raw:
            return None
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return raw
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()


__all__ = ["GoogleNewsCollector"]
