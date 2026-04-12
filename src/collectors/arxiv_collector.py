from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

from src.utils.logger import logger


class ArxivCollector:
    def __init__(self):
        self.base_url = "https://export.arxiv.org/api/query"

    def fetch_latest_ai_papers(self, limit: int = 20) -> list[dict]:
        logger.info("Fetching top %s targeted AI papers from arXiv...", limit)
        query = (
            "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV) AND "
            '(all:"agent" OR all:"reasoning" OR all:"foundation model" OR all:"multimodal")'
        )
        url = (
            f"{self.base_url}?search_query={query}"
            f"&sortBy=submittedDate&sortOrder=descending&max_results={limit}"
        )
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - live source failures
            logger.error("arXiv fetch failed: %s", exc)
            return []

        root = ET.fromstring(response.text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        results: list[dict] = []
        for entry in root.findall("atom:entry", namespace):
            title_node = entry.find("atom:title", namespace)
            summary_node = entry.find("atom:summary", namespace)
            link_node = entry.find("atom:id", namespace)
            published_node = entry.find("atom:published", namespace)
            if None in {title_node, summary_node, link_node, published_node}:
                continue
            authors = ", ".join(
                author.find("atom:name", namespace).text
                for author in entry.findall("atom:author", namespace)[:3]
                if author.find("atom:name", namespace) is not None
            )
            results.append(
                {
                    "title": f"[arXiv] {title_node.text.strip().replace('\n', ' ')}",
                    "url": link_node.text,
                    "desc": (
                        f"Authors: {authors or 'N/A'} | Published: {published_node.text[:10]} | "
                        f"Abstract: {summary_node.text.strip().replace('\n', ' ')[:250]}..."
                    ),
                    "source_name": "arXiv",
                    "source_type": "research",
                    "published_at": published_node.text,
                }
            )
        return results

    def fetch_all_arxiv(self) -> list[dict]:
        return self.fetch_latest_ai_papers()
