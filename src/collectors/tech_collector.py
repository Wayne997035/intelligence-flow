from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

from src.config import Config
from src.utils.logger import logger

try:
    import feedparser
except ImportError:  # pragma: no cover - optional in minimal env
    feedparser = None


class TechCollector:
    def __init__(self):
        self.github_token = Config.GITHUB_TOKEN
        self.session = requests.Session()
        self.primary_focus = [
            "Claude",
            "Gemini",
            "Gemma",
            "OpenAI",
            "ChatGPT",
            "Codex",
            "xAI",
            "Grok",
            "GPT",
            "DeepSeek",
            "Llama",
            "Mistral",
            "Qwen",
            "RAG",
            "AI Agent",
            "Agentic",
            "Skill",
            "VLM",
            "BitNet",
        ]

    def fetch_hacker_news_ai(self) -> list[dict]:
        logger.info("Fetching HN AI stories...")
        query = " OR ".join(f'"{keyword}"' for keyword in self.primary_focus)
        url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage=10"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - live source failures
            logger.error("HN fetch failed: %s", exc)
            return []

        results: list[dict] = []
        for hit in payload.get("hits", []):
            title = hit.get("title")
            if not title:
                continue
            results.append(
                {
                    "title": f"[HN] {title}",
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    "desc": f"Points: {hit.get('points', 0)} | Comments: {hit.get('num_comments', 0)}",
                    "source_name": "Hacker News",
                    "source_type": "community",
                    "published_at": hit.get("created_at"),
                }
            )
        return results

    def fetch_github_trending_ai(self) -> list[dict]:
        logger.info("Fetching GitHub AI trends...")
        since_dt = datetime.now(timezone.utc) - timedelta(days=7)
        last_week = since_dt.strftime("%Y-%m-%d")
        queries = [
            f"topic:llm stars:>150 pushed:>{last_week}",
            f"topic:generative-ai stars:>100 pushed:>{last_week}",
            f"topic:ai-agent stars:>40 pushed:>{last_week}",
            f"topic:rag stars:>40 pushed:>{last_week}",
            f"agent in:name,description stars:>80 pushed:>{last_week}",
            f"skill in:name,description stars:>50 pushed:>{last_week}",
        ]
        headers = self._github_headers()

        results_by_url: dict[str, dict] = {}
        for query in queries:
            try:
                response = self.session.get(
                    "https://api.github.com/search/repositories",
                    headers=headers,
                    params={"q": query, "sort": "stars", "order": "desc", "per_page": 5},
                    timeout=10,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # pragma: no cover - live source failures
                logger.error("GitHub fetch failed for query %s: %s", query, exc)
                continue

            for repo in payload.get("items", []):
                repo_payload = {
                    "full_name": repo["full_name"],
                    "url": repo["html_url"],
                    "stars_total": repo.get("stargazers_count", 0),
                    "desc": repo.get("description") or "No description.",
                    "source_name": "GitHub",
                    "source_type": "github_repo",
                    "published_at": repo.get("pushed_at") or repo.get("updated_at") or repo.get("created_at"),
                }
                existing = results_by_url.get(repo_payload["url"])
                if existing is None or len(repo_payload["desc"]) > len(existing["desc"]):
                    results_by_url[repo_payload["url"]] = repo_payload

        results = list(results_by_url.values())
        for repo_payload in results:
            repo_payload["recent_star_delta"] = self._estimate_recent_star_delta(
                repo_payload["full_name"],
                since_dt,
            )
            delta = repo_payload["recent_star_delta"]
            stars_total = repo_payload["stars_total"]
            if isinstance(delta, int) and delta > 0:
                repo_payload["title"] = f"[GitHub] {repo_payload['full_name']} (+{delta}★/7d, {stars_total} stars)"
            else:
                repo_payload["title"] = f"[GitHub] {repo_payload['full_name']} ({stars_total} stars)"

        def _published_ts(value: str | None) -> float:
            if not value:
                return float("-inf")
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
            except ValueError:
                return float("-inf")

        results.sort(
            key=lambda item: (
                item.get("recent_star_delta") if isinstance(item.get("recent_star_delta"), int) else -1,
                _published_ts(item.get("published_at")),
                item.get("stars_total", 0),
            ),
            reverse=True,
        )
        for repo_payload in results:
            repo_payload.pop("full_name", None)
            repo_payload.pop("stars_total", None)
            repo_payload.pop("recent_star_delta", None)
        return results

    def fetch_reddit_ai_hot(self) -> list[dict]:
        logger.info("Fetching Reddit AI posts...")
        subreddits = ["ClaudeAI", "OpenAI", "GoogleGemini", "singularity", "LocalLLaMA", "MachineLearning"]
        results: list[dict] = []

        for subreddit in subreddits:
            rss_items = self._fetch_reddit_rss(subreddit)
            if rss_items:
                results.extend(rss_items)
                continue
            logger.warning("Reddit RSS fetch returned no entries for %s.", subreddit)
        results.extend(self.fetch_reddit_keyword_matches())
        return results

    def fetch_all_community_ai(self) -> list[dict]:
        return self.fetch_hacker_news_ai() + self.fetch_github_trending_ai() + self.fetch_reddit_ai_hot()

    def _fetch_reddit_rss(self, subreddit: str) -> list[dict]:
        if feedparser is None:
            return []

        feed_url = f"https://www.reddit.com/r/{subreddit}/top/.rss?t=day"
        try:
            feed = feedparser.parse(feed_url)
        except Exception as exc:  # pragma: no cover - parser edge cases
            logger.warning("Reddit RSS parse failed for %s: %s", subreddit, exc)
            return []

        results: list[dict] = []
        for entry in getattr(feed, "entries", [])[:5]:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            if not title or not link:
                continue
            summary = (entry.get("summary") or "").strip()
            results.append(
                {
                    "title": f"[Reddit r/{subreddit}] {title}",
                    "url": link,
                    "desc": f"RSS fallback | {summary[:100]}...",
                    "source_name": f"Reddit/{subreddit}",
                    "source_type": "community",
                    "published_at": entry.get("published") or entry.get("updated"),
                }
            )
        return results

    def fetch_reddit_keyword_matches(self) -> list[dict]:
        if feedparser is None:
            return []

        queries = [
            ("ClaudeAI", "ultraplan"),
            ("ClaudeAI", "\"claude code\""),
            ("OpenAI", "\"responses api\""),
            ("GoogleGemini", "\"gemini api\""),
            ("singularity", "\"grok\""),
        ]
        results: list[dict] = []
        seen_links: set[str] = set()

        for subreddit, query in queries:
            feed_url = (
                f"https://www.reddit.com/r/{subreddit}/search.rss?q={quote(query)}"
                "&restrict_sr=on&sort=new&t=week"
            )
            try:
                feed = feedparser.parse(feed_url)
            except Exception as exc:  # pragma: no cover - parser edge cases
                logger.warning("Reddit search RSS parse failed for %s / %s: %s", subreddit, query, exc)
                continue

            for entry in getattr(feed, "entries", [])[:3]:
                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()
                if not title or not link or link in seen_links:
                    continue
                seen_links.add(link)
                summary = (entry.get("summary") or "").strip()
                recent_feature_signal = self._has_recent_feature_signal(title, summary)
                results.append(
                    {
                        "title": f"[Reddit r/{subreddit}] {title}",
                        "url": link,
                        "desc": f"Search match for {query} | {summary[:120]}...",
                        "source_name": f"Reddit/{subreddit}",
                        "source_type": "community",
                        "published_at": entry.get("published") or entry.get("updated"),
                        "metadata": {
                            "keyword_match": query,
                            "recent_feature_signal": recent_feature_signal,
                        },
                    }
                )
        return results

    def _has_recent_feature_signal(self, title: str, summary: str) -> bool:
        text = f"{title} {summary}".lower()
        signal_phrases = [
            "introduces",
            "introduced",
            "now available",
            "rolling out",
            "rollout",
            "released",
            "release",
            "preview",
            "draft",
            "new feature",
            "new mode",
            "got",
            "to work",
            "feels more like",
            "first impressions",
            "just shipped",
        ]
        return any(phrase in text for phrase in signal_phrases)

    def _github_headers(self, *, timeline_preview: bool = False) -> dict[str, str]:
        accept = "application/vnd.github+json"
        if timeline_preview:
            accept = "application/vnd.github.star+json"
        headers = {
            "Accept": accept,
            "User-Agent": "Intel-Flow-Bot",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    def _estimate_recent_star_delta(self, full_name: str, since_dt: datetime, max_pages: int = 3) -> int | None:
        headers = self._github_headers(timeline_preview=True)
        recent_stars = 0

        for page in range(1, max_pages + 1):
            try:
                response = self.session.get(
                    f"https://api.github.com/repos/{full_name}/stargazers",
                    headers=headers,
                    params={"per_page": 100, "page": page},
                    timeout=10,
                )
                if response.status_code == 403 and "rate limit" in response.text.lower():
                    logger.warning("GitHub stargazer timeline hit rate limit for %s.", full_name)
                    return None
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # pragma: no cover - live source failures
                logger.warning("GitHub stargazer timeline fetch failed for %s: %s", full_name, exc)
                return None

            if not isinstance(payload, list) or not payload:
                break

            reached_older_stars = False
            for entry in payload:
                starred_at = entry.get("starred_at")
                if not starred_at:
                    # If preview header stops returning timeline timestamps, fall back to total stars only.
                    return None
                try:
                    starred_at_dt = datetime.fromisoformat(starred_at.replace("Z", "+00:00")).astimezone(timezone.utc)
                except ValueError:
                    continue
                if starred_at_dt >= since_dt:
                    recent_stars += 1
                else:
                    reached_older_stars = True

            if reached_older_stars or len(payload) < 100:
                break

        return recent_stars
