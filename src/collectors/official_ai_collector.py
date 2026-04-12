from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    import feedparser
except ImportError:  # pragma: no cover - optional in minimal test envs
    feedparser = None

from src.utils.logger import logger


class OfficialAICollector:
    def __init__(self):
        self.feed_sources = [
            {
                "name": "GitHub Blog",
                "url": "https://github.blog/feed/",
                "keywords": ["copilot", "agent", "ai", "models", "github"],
            },
            {
                "name": "Google Developers Blog",
                "url": "https://developers.googleblog.com/feeds/posts/default?alt=rss",
                "keywords": ["gemini", "gemma", "ai", "model", "agent", "vertex", "developer"],
            },
            {
                "name": "OpenAI News RSS",
                "url": "https://openai.com/news/rss.xml",
                "keywords": [
                    "openai",
                    "gpt",
                    "api",
                    "model",
                    "release",
                    "agent",
                    "codex",
                    "chatgpt",
                    "responses",
                ],
            },
        ]
        self.html_sources = [
            {
                "name": "Anthropic News",
                "url": "https://www.anthropic.com/news",
                "link_prefixes": ["/news/"],
                "keywords": [
                    "claude",
                    "anthropic",
                    "model",
                    "agent",
                    "managed agents",
                    "api",
                    "sonnet",
                    "opus",
                ],
            },
        ]

    def fetch_updates(self, limit_per_source: int = 4) -> list[dict]:
        return (
            self._fetch_feed_updates(limit_per_source)
            + self._fetch_html_updates(limit_per_source)
            + self._fetch_claude_platform_release_notes(limit_per_source)
        )

    def _fetch_feed_updates(self, limit_per_source: int) -> list[dict]:
        if feedparser is None:
            logger.warning("feedparser not installed; skipping official AI feeds.")
            return []

        results: list[dict] = []
        for source in self.feed_sources:
            results.extend(
                self._fetch_single_feed_source(
                    source_name=source["name"],
                    url=source["url"],
                    keywords=source["keywords"],
                    limit=limit_per_source,
                )
            )
        return results

    def _fetch_single_feed_source(
        self,
        *,
        source_name: str,
        url: str,
        keywords: list[str],
        limit: int,
    ) -> list[dict]:
        if feedparser is None:
            return []

        results: list[dict] = []
        try:
            feed = feedparser.parse(url)
        except Exception as exc:  # pragma: no cover - feed parser edge cases
            logger.warning("Official AI feed parse failed for %s: %s", source_name, exc)
            return []

        count = 0
        for entry in getattr(feed, "entries", []):
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            summary = (entry.get("summary") or entry.get("description") or "").strip()
            if not title or not link:
                continue
            if not self._matches_keywords(title, summary, keywords):
                continue
            results.append(
                {
                    "title": f"[Official] {title}",
                    "url": link,
                    "desc": BeautifulSoup(summary[:220], "html.parser").get_text(" ", strip=True),
                    "source_name": source_name,
                    "source_type": "official_news",
                    "published_at": self._extract_published_at(entry),
                }
            )
            count += 1
            if count >= limit:
                break
        return results

    def _fetch_html_updates(self, limit_per_source: int) -> list[dict]:
        results: list[dict] = []
        generic_titles = {"research", "product", "company", "safety", "news", "index"}
        for source in self.html_sources:
            try:
                response = requests.get(source["url"], timeout=10, headers={"User-Agent": "Intel-Flow-Bot"})
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - live source failures
                fallback_feed_url = source.get("fallback_feed_url")
                if fallback_feed_url:
                    logger.info(
                        "Official AI source %s is using RSS fallback: %s",
                        source["name"],
                        fallback_feed_url,
                    )
                    results.extend(
                        self._fetch_single_feed_source(
                            source_name=source.get("fallback_feed_source_name", source["name"]),
                            url=fallback_feed_url,
                            keywords=source.get("fallback_feed_keywords", source["keywords"]),
                            limit=limit_per_source,
                        )
                    )
                    continue
                logger.warning("Official AI page fetch failed for %s: %s", source["name"], exc)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            seen_links: set[str] = set()
            source_items: list[dict] = []
            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()
                if not any(prefix in href for prefix in source["link_prefixes"]):
                    continue

                title = anchor.get_text(" ", strip=True)
                if (
                    not title
                    or len(title) < 16
                    or title.lower() in generic_titles
                    or not self._matches_keywords(title, "", source["keywords"])
                ):
                    continue

                absolute_url = urljoin(source["url"], href)
                if absolute_url in seen_links:
                    continue
                seen_links.add(absolute_url)
                published_at = self._extract_html_published_at(anchor)
                if not published_at:
                    continue

                source_items.append(
                    {
                        "title": f"[Official] {title}",
                        "url": absolute_url,
                        "desc": f"Collected from {source['name']} listing page.",
                        "source_name": source["name"],
                        "source_type": "official_news",
                        "published_at": published_at,
                    }
                )
            source_items.sort(key=lambda item: item.get("published_at") or "", reverse=True)
            results.extend(source_items[:limit_per_source])
        return results

    def _fetch_claude_platform_release_notes(self, limit: int) -> list[dict]:
        url = "https://platform.claude.com/docs/en/release-notes/overview"
        keywords = ["managed agents", "advisor", "tool", "agent", "api", "model", "sdk", "release", "claude"]
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "Intel-Flow-Bot"})
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - live source failures
            logger.warning("Claude Platform release-notes fetch failed: %s", exc)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results_by_url: dict[str, dict] = {}

        # Primary extraction path: date headings + adjacent bullet lists.
        for heading in soup.find_all(["h2", "h3", "h4"]):
            heading_text = heading.get_text(" ", strip=True)
            published_at = self._normalize_datetime(heading_text)
            if not published_at:
                continue

            heading_id = heading.get("id")
            note_url = f"{url}#{heading_id}" if heading_id else url

            bullet_items: list[tuple[str, str]] = []
            node = heading.find_next_sibling()
            while node and node.name not in {"h2", "h3", "h4"}:
                if node.name in {"ul", "ol"}:
                    for li in node.find_all("li"):
                        text = li.get_text(" ", strip=True)
                        if not text or not self._matches_keywords(text, "", keywords):
                            continue
                        anchor = li.find("a", href=True)
                        linked_url = urljoin(url, anchor["href"]) if anchor else note_url
                        bullet_items.append((text, linked_url))
                node = node.find_next_sibling()

            if not bullet_items:
                continue

            for bullet_text, bullet_url in bullet_items:
                title = f"[Official] Claude Platform {heading_text}: {bullet_text[:96]}"
                existing = results_by_url.get(bullet_url)
                candidate = {
                    "title": title,
                    "url": bullet_url,
                    "desc": bullet_text[:220],
                    "source_name": "Claude Platform Release Notes",
                    "source_type": "official_news",
                    "published_at": published_at,
                }
                if existing is None or len(candidate["desc"]) > len(existing.get("desc", "")):
                    results_by_url[bullet_url] = candidate

        # Fallback extraction path: keyword-heavy links in release-notes docs if DOM shape changes.
        for anchor in soup.find_all("a", href=True):
            href = (anchor.get("href") or "").strip()
            text = anchor.get_text(" ", strip=True)
            if not text or len(text) < 24:
                continue
            if not self._matches_keywords(text, "", keywords):
                continue

            absolute_url = urljoin(url, href)
            nearby_heading = anchor.find_previous(["h2", "h3", "h4"])
            heading_text = nearby_heading.get_text(" ", strip=True) if nearby_heading else "Update"
            published_at = (
                self._normalize_datetime(heading_text)
                or self._extract_html_published_at(anchor)
                or self._normalize_datetime(self._extract_date_from_text(anchor.parent.get_text(" ", strip=True)))
            )
            if not published_at:
                continue

            context = anchor.parent.get_text(" ", strip=True)
            desc = (context or text)[:220]
            title = f"[Official] Claude Platform {heading_text}: {text[:96]}"
            existing = results_by_url.get(absolute_url)
            candidate = {
                "title": title,
                "url": absolute_url,
                "desc": desc,
                "source_name": "Claude Platform Release Notes",
                "source_type": "official_news",
                "published_at": published_at,
            }
            if existing is None or len(candidate["desc"]) > len(existing.get("desc", "")):
                results_by_url[absolute_url] = candidate

        results = list(results_by_url.values())
        results.sort(key=lambda item: item.get("published_at") or "", reverse=True)
        return results[:limit]

    def _matches_keywords(self, title: str, summary: str, keywords: list[str]) -> bool:
        text = f"{title} {summary}".lower()
        return any(keyword.lower() in text for keyword in keywords)

    def _extract_published_at(self, entry) -> str | None:
        if entry.get("published"):
            try:
                return parsedate_to_datetime(entry["published"]).astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError):
                return entry["published"]
        if entry.get("updated"):
            try:
                return parsedate_to_datetime(entry["updated"]).astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError):
                return entry["updated"]
        if getattr(entry, "published_parsed", None):
            parsed = entry.published_parsed
            return datetime(
                parsed.tm_year,
                parsed.tm_mon,
                parsed.tm_mday,
                parsed.tm_hour,
                parsed.tm_min,
                parsed.tm_sec,
                tzinfo=timezone.utc,
            ).isoformat()
        if getattr(entry, "updated_parsed", None):
            parsed = entry.updated_parsed
            return datetime(
                parsed.tm_year,
                parsed.tm_mon,
                parsed.tm_mday,
                parsed.tm_hour,
                parsed.tm_min,
                parsed.tm_sec,
                tzinfo=timezone.utc,
            ).isoformat()
        return None

    def _extract_html_published_at(self, anchor) -> str | None:
        candidates: list[str] = []
        attr_names = ("datetime", "data-date", "data-published", "data-published-at")
        nearby_nodes = [anchor, *list(anchor.parents)[:3]]

        for node in nearby_nodes:
            for attr_name in attr_names:
                value = node.get(attr_name)
                if value:
                    candidates.append(value)

            time_node = node.find("time")
            if time_node:
                datetime_value = time_node.get("datetime")
                if datetime_value:
                    candidates.append(datetime_value)
                time_text = time_node.get_text(" ", strip=True)
                if time_text:
                    candidates.append(time_text)

            text_candidate = self._extract_date_from_text(node.get_text(" ", strip=True))
            if text_candidate:
                candidates.append(text_candidate)

        for candidate in candidates:
            normalized = self._normalize_datetime(candidate)
            if normalized:
                return normalized
        return None

    def _extract_date_from_text(self, text: str) -> str | None:
        if not text:
            return None
        patterns = [
            r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+20\d{2}\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _normalize_datetime(self, value: str) -> str | None:
        cleaned = (value or "").strip()
        if not cleaned:
            return None

        parsed: datetime | None = None
        if cleaned.endswith("Z"):
            cleaned = f"{cleaned[:-1]}+00:00"

        try:
            parsed = parsedate_to_datetime(cleaned)
        except (TypeError, ValueError):
            pass

        if parsed is None:
            try:
                parsed = datetime.fromisoformat(cleaned)
            except ValueError:
                pass

        if parsed is None:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"):
                try:
                    parsed = datetime.strptime(cleaned, fmt)
                    break
                except ValueError:
                    continue

        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
