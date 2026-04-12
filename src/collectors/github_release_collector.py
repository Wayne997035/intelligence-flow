from __future__ import annotations

import requests

from src.config import Config
from src.utils.logger import logger


class GitHubReleaseCollector:
    def __init__(self):
        self.repos = Config.AI_GITHUB_RELEASE_REPOS
        self.github_token = Config.GITHUB_TOKEN

    def fetch_latest_releases(self, per_repo_limit: int = 1) -> list[dict]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Intel-Flow-Bot",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        results: list[dict] = []
        for repo in self.repos:
            try:
                response = requests.get(
                    f"https://api.github.com/repos/{repo}/releases",
                    headers=headers,
                    params={"per_page": per_repo_limit},
                    timeout=10,
                )
                if response.status_code == 403 and "rate limit" in response.text.lower():
                    logger.warning("GitHub release fetch hit rate limit; skipping remaining release checks.")
                    break
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # pragma: no cover - live source failures
                logger.warning("GitHub release fetch failed for %s: %s", repo, exc)
                continue

            for release in payload[:per_repo_limit]:
                results.append(
                    {
                        "title": f"[GitHub Release] {repo} {release.get('name') or release.get('tag_name')}",
                        "url": release.get("html_url", f"https://github.com/{repo}/releases"),
                        "desc": (
                            f"Tag: {release.get('tag_name', 'N/A')} | "
                            f"Published: {release.get('published_at', 'N/A')} | "
                            f"{(release.get('body') or '').strip()[:180]}..."
                        ),
                        "source_name": "GitHub Releases",
                        "source_type": "github_release",
                        "published_at": release.get("published_at"),
                        "tags": [repo],
                    }
                )
        return results
