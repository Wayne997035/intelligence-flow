from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.models import IntelligenceItem

_DROP_QUERY_KEYS = {"ref", "source", "si", "fbclid", "gclid"}
_SOURCE_TYPE_SCORES = {
    "official_news": 100,
    "model_release": 95,
    "research": 90,
    "github_release": 85,
    "github_repo": 80,
    "news": 70,
    "community": 40,
    "unknown": 10,
}


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def canonicalize_url(url: str | None) -> str:
    cleaned = normalize_text(url)
    if not cleaned:
        return ""

    parts = urlsplit(cleaned)
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.startswith("utm_") and key not in _DROP_QUERY_KEYS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (
            (parts.scheme or "https").lower(),
            parts.netloc.lower(),
            path,
            urlencode(query_pairs, doseq=True),
            "",
        )
    )


def infer_tags(text: str, priority_keywords: list[str]) -> list[str]:
    content = normalize_text(text).lower()
    return [keyword for keyword in priority_keywords if keyword.lower() in content]


def normalize_item(
    item: IntelligenceItem | dict,
    *,
    default_source_name: str = "unknown",
    default_source_type: str = "unknown",
    priority_keywords: list[str] | None = None,
) -> IntelligenceItem:
    if isinstance(item, IntelligenceItem):
        normalized = item
    else:
        normalized = IntelligenceItem(
            title=normalize_text(item.get("title")),
            url=canonicalize_url(item.get("url")),
            desc=normalize_text(item.get("desc")),
            source_name=normalize_text(item.get("source_name")) or default_source_name,
            source_type=normalize_text(item.get("source_type")) or default_source_type,
            published_at=normalize_text(item.get("published_at")) or None,
            tags=list(item.get("tags", [])),
            metadata=dict(item.get("metadata", {})),
        )

    if not normalized.tags and priority_keywords:
        normalized.tags = infer_tags(
            f"{normalized.title} {normalized.desc}",
            priority_keywords,
        )

    if priority_keywords:
        normalized.priority = next(
            (
                index
                for index, keyword in enumerate(priority_keywords)
                if keyword.lower() in f"{normalized.title} {normalized.desc}".lower()
            ),
            999,
        )
    return normalized


def is_low_signal_item(item: IntelligenceItem) -> bool:
    title = normalize_text(item.title)
    lowered = title.lower()

    # Filter out very short or generic community headlines.
    generic_titles = {"unexpected", "thoughts", "help", "question", "thought", "wow"}
    if item.source_type == "community":
        if len(title) < 18:
            return True
        if lowered in generic_titles or lowered.startswith("[reddit") and len(title.split()) <= 3:
            return True

    return False


def is_relevant_ai_item(item: IntelligenceItem) -> bool:
    if item.source_type in {"official_news", "model_release", "research", "github_release", "github_repo"}:
        return True

    text = f"{normalize_text(item.title)} {normalize_text(item.desc)}".lower()
    positive_keywords = [
        "model",
        "release",
        "launch",
        "agent",
        "api",
        "sdk",
        "benchmark",
        "reasoning",
        "multimodal",
        "paper",
        "arxiv",
        "github",
        "hugging face",
        "llm",
        "open source",
        "copilot",
        "transformer",
        "claude",
        "gpt",
        "gemini",
        "gemma",
        "deepseek",
        "qwen",
        "llama",
        "mistral",
    ]
    negative_keywords = [
        "lawsuit",
        "probe",
        "investigation",
        "arrested",
        "molotov",
        "shooting",
        "abuser",
        "house",
        "sues",
        "crime",
        "victim",
        "war",
        "department of war",
    ]

    if any(keyword in text for keyword in negative_keywords):
        return False
    return any(keyword in text for keyword in positive_keywords)


def deduplicate_and_rank(
    items: Iterable[IntelligenceItem | dict],
    priority_keywords: list[str],
    limit: int,
    *,
    default_source_name: str = "unknown",
    default_source_type: str = "unknown",
) -> list[IntelligenceItem]:
    unique_items: dict[str, IntelligenceItem] = {}
    title_index: dict[str, str] = {}

    for item in items:
        normalized = normalize_item(
            item,
            default_source_name=default_source_name,
            default_source_type=default_source_type,
            priority_keywords=priority_keywords,
        )
        if not normalized.url or not normalized.title:
            continue
        if is_low_signal_item(normalized):
            continue

        title_key = re.sub(r"[^a-z0-9]+", "", normalized.title.lower())[:96]
        dedupe_key = normalized.url
        if title_key and len(title_key) >= 24:
            dedupe_key = title_index.get(title_key, dedupe_key)

        existing = unique_items.get(dedupe_key)

        if _should_replace(existing, normalized):
            unique_items[dedupe_key] = normalized
            if title_key and len(title_key) >= 24:
                title_index[title_key] = dedupe_key

    def _published_timestamp(item: IntelligenceItem) -> float:
        parsed = parse_published_at(item.published_at)
        if parsed is None:
            return float("-inf")
        return parsed.timestamp()

    ranked = list(unique_items.values())
    ranked.sort(
        key=lambda item: (
            item.priority,
            -source_quality_score(item),
            -_published_timestamp(item),
            item.title.lower(),
        )
    )
    return ranked[:limit]


def parse_published_at(value: str | None) -> datetime | None:
    cleaned = normalize_text(value)
    if not cleaned:
        return None

    if cleaned.endswith("Z"):
        cleaned = f"{cleaned[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        parsed = None

    if parsed is None:
        try:
            parsed = parsedate_to_datetime(cleaned)
        except (TypeError, ValueError):
            parsed = None

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
    return parsed.astimezone(timezone.utc)


def filter_recent_items(
    items: Iterable[IntelligenceItem],
    *,
    max_age_days: int,
    now: datetime | None = None,
    require_published_at: bool = False,
) -> list[IntelligenceItem]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = current - timedelta(days=max_age_days)
    filtered: list[IntelligenceItem] = []

    for item in items:
        published_dt = parse_published_at(item.published_at)
        if published_dt is None:
            if require_published_at:
                continue
            filtered.append(item)
            continue
        if published_dt >= cutoff:
            filtered.append(item)

    return filtered


def summarize_sources(items: Iterable[IntelligenceItem]) -> str:
    counter = Counter(item.source_type for item in items if item.source_type)
    if not counter:
        return "未提供來源分類"
    return "、".join(f"{name} {count} 則" for name, count in counter.most_common())


def source_quality_score(item: IntelligenceItem) -> int:
    return _SOURCE_TYPE_SCORES.get(item.source_type, _SOURCE_TYPE_SCORES["unknown"])


def _should_replace(existing: IntelligenceItem | None, candidate: IntelligenceItem) -> bool:
    if existing is None:
        return True

    existing_score = source_quality_score(existing)
    candidate_score = source_quality_score(candidate)
    if candidate_score != existing_score:
        return candidate_score > existing_score

    if len(candidate.desc) != len(existing.desc):
        return len(candidate.desc) > len(existing.desc)

    return (candidate.published_at or "") > (existing.published_at or "")
