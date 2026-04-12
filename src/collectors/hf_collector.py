from __future__ import annotations

import requests

from src.utils.logger import logger


class HFCollector:
    def __init__(self):
        self.base_url = "https://huggingface.co/api"

    def fetch_trending_models(self, limit: int = 15) -> list[dict]:
        logger.info("Fetching top %s trending models from Hugging Face...", limit)
        url = f"https://huggingface.co/api/trending?type=model&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - live source failures
            logger.error("HF models fetch failed: %s", exc)
            return []

        results: list[dict] = []
        for model in payload.get("recentlyTrending", []):
            repo_data = model.get("repoData", {})
            model_id = repo_data.get("id")
            if not model_id:
                continue
            results.append(
                {
                    "title": f"[HF Model] {model_id} ({repo_data.get('likes', 0)} likes)",
                    "url": f"https://huggingface.co/{model_id}",
                    "desc": f"Downloads: {repo_data.get('downloads', 0)} | Task: {repo_data.get('pipeline_tag', 'N/A')}",
                    "source_name": "Hugging Face",
                    "source_type": "model_release",
                    "published_at": repo_data.get("lastModified"),
                }
            )
        return results

    def fetch_daily_papers(self, limit: int = 20) -> list[dict]:
        logger.info("Fetching top %s daily papers from Hugging Face...", limit)
        url = f"{self.base_url}/daily_papers?limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - live source failures
            logger.error("HF papers fetch failed: %s", exc)
            return []

        results: list[dict] = []
        for paper_item in payload:
            paper = paper_item.get("paper", {})
            title = paper.get("title")
            paper_id = paper.get("id")
            if not title or not paper_id:
                continue
            results.append(
                {
                    "title": f"[HF Paper] {title}",
                    "url": f"https://huggingface.co/papers/{paper_id}",
                    "desc": f"Published: {paper.get('publishedAt', 'N/A')} | Summary: {(paper.get('summary') or '')[:180]}...",
                    "source_name": "Hugging Face",
                    "source_type": "research",
                    "published_at": paper.get("publishedAt"),
                }
            )
        return results

    def fetch_all_hf(self) -> list[dict]:
        return self.fetch_trending_models() + self.fetch_daily_papers()
