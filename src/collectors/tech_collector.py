from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

from src.config import Config
from src.utils.logger import logger


class TechCollector:
    def __init__(self):
        self.github_token = Config.GITHUB_TOKEN
        self.primary_focus = [
            "Claude",
            "Gemini",
            "OpenAI",
            "GPT",
            "DeepSeek",
            "Llama",
            "Mistral",
            "Qwen",
            "RAG",
            "AI Agent",
            "VLM",
            "Grok",
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
        last_week = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        queries = [
            f"topic:llm stars:>200 pushed:>{last_week}",
            f"topic:ai-agent stars:>50 pushed:>{last_week}",
            f"topic:rag stars:>50 pushed:>{last_week}",
        ]
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Intel-Flow-Bot",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        results_by_url: dict[str, dict] = {}
        for query in queries:
            try:
                response = requests.get(
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
                    "title": f"[GitHub] {repo['full_name']} ({repo['stargazers_count']} stars)",
                    "url": repo["html_url"],
                    "desc": repo.get("description") or "No description.",
                    "source_name": "GitHub",
                    "source_type": "github_repo",
                    "published_at": repo.get("pushed_at") or repo.get("updated_at") or repo.get("created_at"),
                }
                existing = results_by_url.get(repo_payload["url"])
                if existing is None or len(repo_payload["desc"]) > len(existing["desc"]):
                    results_by_url[repo_payload["url"]] = repo_payload

        results = list(results_by_url.values())
        results.sort(key=lambda item: item.get("published_at") or "", reverse=True)
        return results

    def fetch_reddit_ai_hot(self) -> list[dict]:
        logger.info("Fetching Reddit AI posts...")
        subreddits = ["ClaudeAI", "OpenAI", "GoogleGemini", "singularity", "LocalLLaMA", "MachineLearning"]
        headers = {"User-Agent": "Intel-Flow-Bot/1.0"}
        results: list[dict] = []

        for subreddit in subreddits:
            try:
                response = requests.get(
                    f"https://www.reddit.com/r/{subreddit}/top.json?t=day&limit=5",
                    headers=headers,
                    timeout=10,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # pragma: no cover - live source failures
                logger.error("Reddit fetch failed for %s: %s", subreddit, exc)
                continue

            for post in payload.get("data", {}).get("children", []):
                data = post.get("data", {})
                if data.get("stickied") or not data.get("title"):
                    continue
                results.append(
                    {
                        "title": f"[Reddit r/{subreddit}] {data['title']}",
                        "url": f"https://www.reddit.com{data['permalink']}",
                        "desc": f"Upvotes: {data.get('ups', 0)} | {data.get('selftext', '')[:100]}...",
                        "source_name": f"Reddit/{subreddit}",
                        "source_type": "community",
                        "published_at": datetime.fromtimestamp(
                            data.get("created_utc", 0),
                            tz=timezone.utc,
                        ).isoformat(),
                    }
                )
        return results

    def fetch_all_community_ai(self) -> list[dict]:
        return self.fetch_hacker_news_ai() + self.fetch_github_trending_ai() + self.fetch_reddit_ai_hot()
