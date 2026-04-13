from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover - optional in minimal test envs
    curl_requests = None

try:
    import feedparser
except ImportError:  # pragma: no cover - optional in minimal test envs
    feedparser = None

from src.utils.logger import logger


class OfficialAICollector:
    _BASE_BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

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
            {
                "name": "Google Blog Gemini App",
                "url": "https://blog.google/innovation-and-ai/products/gemini-app/",
                "link_prefixes": ["/innovation-and-ai/products/gemini-app/"],
                "keywords": [
                    "gemini",
                    "notebooklm",
                    "notebooks",
                    "deep research",
                    "gemini app",
                    "gemini live",
                    "project",
                    "google ai",
                ],
            },
            {
                "name": "Google Blog Gemini",
                "url": "https://blog.google/products/gemini/",
                "link_prefixes": ["/products/gemini/", "/products-and-platforms/products/gemini/"],
                "keywords": [
                    "gemini",
                    "notebooklm",
                    "notebooks",
                    "deep think",
                    "flash",
                    "pro",
                    "gemma",
                    "google ai",
                ],
            },
        ]
        self.docs_sources = [
            {
                "name": "Claude Code Changelog",
                "url": "https://code.claude.com/docs/en/changelog",
                "keywords": [
                    "claude code",
                    "ultraplan",
                    "plan mode",
                    "mcp",
                    "plugin",
                    "agent",
                    "agents",
                    "powerup",
                    "remote",
                ],
                "link_hints": {
                    "ultraplan": "https://code.claude.com/docs/en/ultraplan",
                },
            },
            {
                "name": "OpenAI API Changelog",
                "url": "https://platform.openai.com/docs/changelog",
                "keywords": [
                    "gpt",
                    "responses",
                    "agents",
                    "chatkit",
                    "codex",
                    "builder",
                    "realtime",
                    "reasoning",
                    "tools",
                    "api",
                ],
            },
            {
                "name": "Gemini API Release Notes",
                "url": "https://ai.google.dev/gemini-api/docs/changelog",
                "keywords": [
                    "gemini",
                    "gemma",
                    "agent",
                    "interactions api",
                    "deep research",
                    "tool",
                    "live api",
                    "reasoning",
                    "thinking",
                ],
            },
            {
                "name": "xAI Release Notes",
                "url": "https://docs.x.ai/developers/release-notes",
                "keywords": [
                    "grok",
                    "agent",
                    "multi-agent",
                    "batch api",
                    "live search",
                    "image generation",
                    "voice agent",
                    "api",
                    "tools",
                ],
            },
        ]

    def fetch_updates(self, limit_per_source: int = 4) -> list[dict]:
        return (
            self._fetch_feed_updates(limit_per_source)
            + self._fetch_html_updates(limit_per_source)
            + self._fetch_claude_platform_release_notes(limit_per_source)
            + self._fetch_docs_release_notes(limit_per_source)
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
                html = self._fetch_page_text(source["url"], source_name=source["name"])
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

            soup = BeautifulSoup(html, "html.parser")
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
            html = self._fetch_page_text(url, source_name="Claude Platform Release Notes")
        except Exception as exc:  # pragma: no cover - live source failures
            logger.warning("Claude Platform release-notes fetch failed: %s", exc)
            return []

        soup = BeautifulSoup(html, "html.parser")
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

    def _fetch_docs_release_notes(self, limit_per_source: int) -> list[dict]:
        results: list[dict] = []
        for source in self.docs_sources:
            try:
                html = self._fetch_page_text(source["url"], source_name=source["name"])
            except Exception as exc:  # pragma: no cover - live source failures
                logger.warning("Official docs release-notes fetch failed for %s: %s", source["name"], exc)
                continue

            results.extend(
                self._extract_docs_release_notes(
                    source_name=source["name"],
                    source_url=source["url"],
                    html=html,
                    keywords=source["keywords"],
                    limit=limit_per_source,
                    link_hints=source.get("link_hints"),
                )
            )
        return results

    def _fetch_page_text(self, url: str, *, source_name: str) -> str:
        headers = self._build_browser_headers(url, include_brotli=False)
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            if self._looks_like_html(response.text):
                return response.text
            if curl_requests is None:
                return response.text
        except Exception as exc:
            if not self._is_http_403(exc) or curl_requests is None:
                raise

            logger.info(
                "Retrying %s with curl_cffi browser impersonation after 403.",
                source_name,
            )
            return self._fetch_page_text_with_curl(url, source_name=source_name)

        logger.info(
            "Retrying %s with curl_cffi because requests returned non-HTML content.",
            source_name,
        )
        return self._fetch_page_text_with_curl(url, source_name=source_name)

    def _fetch_page_text_with_curl(self, url: str, *, source_name: str) -> str:
        response = curl_requests.get(
            url,
            timeout=20,
            headers=self._build_browser_headers(url, include_brotli=True),
            impersonate="chrome131",
        )
        response.raise_for_status()
        return response.text

    def _build_browser_headers(self, url: str, *, include_brotli: bool) -> dict[str, str]:
        headers = dict(self._BASE_BROWSER_HEADERS)
        headers["Accept-Encoding"] = "gzip, deflate, br" if include_brotli else "gzip, deflate"
        headers["Referer"] = self._derive_referer(url)
        return headers

    def _derive_referer(self, url: str) -> str:
        if "platform.openai.com" in url:
            return "https://platform.openai.com/"
        if "ai.google.dev" in url:
            return "https://ai.google.dev/"
        if "docs.x.ai" in url:
            return "https://docs.x.ai/"
        if "code.claude.com" in url:
            return "https://code.claude.com/"
        if "platform.claude.com" in url:
            return "https://platform.claude.com/"
        if "blog.google" in url:
            return "https://blog.google/"
        if "anthropic.com" in url:
            return "https://www.anthropic.com/"
        return url

    def _is_http_403(self, exc: Exception) -> bool:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 403:
            return True
        return "403" in str(exc)

    def _looks_like_html(self, text: str) -> bool:
        if not text:
            return False
        prefix = text[:512].lower()
        return "<!doctype html" in prefix or "<html" in prefix or "<head" in prefix

    def _extract_docs_release_notes(
        self,
        *,
        source_name: str,
        source_url: str,
        html: str,
        keywords: list[str],
        limit: int,
        link_hints: dict[str, str] | None = None,
    ) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        if source_name == "OpenAI API Changelog":
            openai_items = self._extract_openai_changelog_cards(
                soup=soup,
                source_name=source_name,
                source_url=source_url,
                keywords=keywords,
                limit=limit,
            )
            if openai_items:
                return openai_items
        results_by_url: dict[str, dict] = {}

        for heading in soup.find_all(["h2", "h3", "h4"]):
            heading_text = heading.get_text(" ", strip=True)
            published_at = self._resolve_section_published_at(heading, heading_text)
            if not published_at:
                continue

            section_anchor = heading.get("id")
            default_url = f"{source_url}#{section_anchor}" if section_anchor else source_url
            bullet_items = self._extract_section_bullets(heading, source_url, link_hints=link_hints)
            for bullet_text, bullet_url in bullet_items:
                if not self._matches_keywords(bullet_text, heading_text, keywords):
                    continue
                candidate = {
                    "title": f"[Official] {source_name} {heading_text}: {bullet_text[:96]}",
                    "url": bullet_url or default_url,
                    "desc": bullet_text[:220],
                    "source_name": source_name,
                    "source_type": "official_news",
                    "published_at": published_at,
                }
                existing = results_by_url.get(candidate["url"])
                if existing is None or len(candidate["desc"]) > len(existing.get("desc", "")):
                    results_by_url[candidate["url"]] = candidate

        # Fallback for docs pages that use dated sections with paragraphs instead of bullet lists.
        for heading in soup.find_all(["h2", "h3", "h4"]):
            heading_text = heading.get_text(" ", strip=True)
            published_at = self._resolve_section_published_at(heading, heading_text)
            if not published_at:
                continue

            section_anchor = heading.get("id")
            default_url = f"{source_url}#{section_anchor}" if section_anchor else source_url
            node = heading.find_next_sibling()
            while node and node.name not in {"h2", "h3", "h4"}:
                if node.name == "p":
                    text = node.get_text(" ", strip=True)
                    if text and self._matches_keywords(text, heading_text, keywords):
                        candidate = {
                            "title": f"[Official] {source_name} {heading_text}: {text[:96]}",
                            "url": default_url,
                            "desc": text[:220],
                            "source_name": source_name,
                            "source_type": "official_news",
                            "published_at": published_at,
                        }
                        existing = results_by_url.get(candidate["url"])
                        if existing is None or len(candidate["desc"]) > len(existing.get("desc", "")):
                            results_by_url[candidate["url"]] = candidate
                node = node.find_next_sibling()

        results = list(results_by_url.values())
        results.sort(key=lambda item: item.get("published_at") or "", reverse=True)
        return results[:limit]

    def _extract_openai_changelog_cards(
        self,
        *,
        soup: BeautifulSoup,
        source_name: str,
        source_url: str,
        keywords: list[str],
        limit: int,
    ) -> list[dict]:
        results: list[dict] = []
        seen_urls: set[str] = set()
        current_year = datetime.now(timezone.utc).year
        previous_month: int | None = None

        for grid in soup.find_all("div", class_=lambda value: value and "grid-cols-[3rem_1fr]" in " ".join(value if isinstance(value, list) else [value])):
            children = [child for child in grid.find_all("div", recursive=False)]
            if len(children) < 2:
                continue

            date_badge = children[0].find("div", attrs={"data-variant": "outline"})
            date_text = date_badge.get_text(" ", strip=True) if date_badge else ""
            month_day = self._parse_openai_month_day(date_text)
            if month_day is None:
                continue

            month_num, day = month_day
            if previous_month is None:
                now_month = datetime.now(timezone.utc).month
                if month_num > now_month + 1:
                    current_year -= 1
            elif month_num > previous_month:
                current_year -= 1
            previous_month = month_num

            published_at = datetime(current_year, month_num, day, tzinfo=timezone.utc).isoformat()
            content = children[1]

            badge_texts = [
                badge.get_text(" ", strip=True)
                for badge in content.find_all("div", attrs={"data-variant": "soft"})
                if badge.get_text(" ", strip=True)
            ]
            markdown = content.find("div", class_=lambda value: value and "ChangelogMarkdown" in " ".join(value if isinstance(value, list) else [value]))
            body_text = markdown.get_text(" ", strip=True) if markdown else content.get_text(" ", strip=True)
            combined_text = " ".join([*badge_texts, body_text]).strip()
            if not self._matches_keywords(combined_text, "", keywords):
                continue

            anchor = markdown.find("a", href=True) if markdown else content.find("a", href=True)
            item_url = urljoin(source_url, anchor["href"]) if anchor else source_url
            if item_url in seen_urls:
                continue
            seen_urls.add(item_url)

            tags = " / ".join(text for text in badge_texts[:3] if text)
            lead = body_text[:96] if body_text else "Update"
            title_parts = [source_name, date_text]
            if tags:
                title_parts.append(tags)
            title = f"[Official] {' | '.join(title_parts)}: {lead}"
            results.append(
                {
                    "title": title,
                    "url": item_url,
                    "desc": body_text[:220] if body_text else lead,
                    "source_name": source_name,
                    "source_type": "official_news",
                    "published_at": published_at,
                }
            )
            if len(results) >= limit:
                break

        return results[:limit]

    def _extract_section_bullets(
        self,
        heading,
        source_url: str,
        *,
        link_hints: dict[str, str] | None = None,
    ) -> list[tuple[str, str]]:
        bullet_items: list[tuple[str, str]] = []
        node = heading.find_next_sibling()
        while node and node.name not in {"h2", "h3", "h4"}:
            if node.name in {"ul", "ol"}:
                for li in node.find_all("li"):
                    text = li.get_text(" ", strip=True)
                    if not text:
                        continue
                    anchor = li.find("a", href=True)
                    linked_url = urljoin(source_url, anchor["href"]) if anchor else self._infer_docs_link(text, source_url, link_hints)
                    bullet_items.append((text, linked_url))
            node = node.find_next_sibling()
        return bullet_items

    def _infer_docs_link(
        self,
        text: str,
        source_url: str,
        link_hints: dict[str, str] | None = None,
    ) -> str:
        normalized = text.lower()
        for keyword, url in (link_hints or {}).items():
            if keyword.lower() in normalized:
                return url
        return source_url

    def _resolve_section_published_at(self, heading, heading_text: str) -> str | None:
        published_at = self._normalize_datetime(heading_text)
        if published_at:
            return published_at

        node = heading.find_next_sibling()
        inspected = 0
        while node and node.name not in {"h2", "h3", "h4"} and inspected < 3:
            text = node.get_text(" ", strip=True)
            candidate = self._extract_date_from_text(text) or text
            published_at = self._normalize_datetime(candidate)
            if published_at:
                return published_at
            inspected += 1
            node = node.find_next_sibling()
        return None

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
        attr_names = (
            "datetime",
            "data-date",
            "data-published",
            "data-published-at",
            "data-ga4-analytics-lead-click",
        )
        nearby_nodes = [anchor, *list(anchor.parents)[:3]]

        for node in nearby_nodes:
            for attr_name in attr_names:
                value = node.get(attr_name)
                if value:
                    candidates.append(value)

            for attr_value in node.attrs.values():
                if isinstance(attr_value, str):
                    candidates.append(attr_value)
                elif isinstance(attr_value, (list, tuple)):
                    candidates.extend(str(value) for value in attr_value if value)

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
            publish_date_match = re.search(r'"publish_date"\s*:\s*"([^"]+)"', candidate)
            if publish_date_match:
                candidate = publish_date_match.group(1).split("|", 1)[0]
            normalized = self._normalize_datetime(candidate)
            if normalized:
                return normalized
            extracted = self._extract_date_from_text(candidate)
            if extracted:
                normalized = self._normalize_datetime(extracted)
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

    def _parse_openai_month_day(self, text: str) -> tuple[int, int] | None:
        match = re.fullmatch(r"([A-Za-z]{3})\s+(\d{1,2})", (text or "").strip())
        if not match:
            return None
        try:
            month = datetime.strptime(match.group(1), "%b").month
        except ValueError:
            return None
        return month, int(match.group(2))

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
