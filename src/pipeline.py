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
_CORE_PROVIDER_TERMS = (
    "claude",
    "anthropic",
    "gpt",
    "chatgpt",
    "openai",
    "codex",
    "gemini",
    "google",
    "xai",
    "grok",
)


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
    fragment = parts.fragment.strip()
    return urlunsplit(
        (
            (parts.scheme or "https").lower(),
            parts.netloc.lower(),
            path,
            urlencode(query_pairs, doseq=True),
            fragment,
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
    desc = normalize_text(item.desc).lower()
    source_name = normalize_text(item.source_name).lower()
    url = canonicalize_url(item.url).lower()

    # Filter out very short or generic community headlines.
    generic_titles = {"unexpected", "thoughts", "help", "question", "thought", "wow"}
    if item.source_type == "community":
        if len(title) < 18:
            return True
        if lowered in generic_titles or lowered.startswith("[reddit") and len(title.split()) <= 3:
            return True

    if item.source_type in {"official_news", "news"}:
        if "openai.com/academy/" in url and not any(
            keyword in f"{lowered} {desc}"
            for keyword in ("responses", "codex", "projects", "connector", "api", "agent builder")
        ):
            return True
        if any(
            keyword in f"{lowered} {desc}"
            for keyword in (
                "marketing teams",
                "study for finals",
                "education",
                "for beginners",
                "tips",
                "how to use",
            )
        ) and not any(keyword in f"{lowered} {desc}" for keyword in ("notebooklm", "notebooks", "codex", "responses")):
            return True

    if item.source_type == "model_release":
        if source_name == "hugging face" and any(keyword in lowered for keyword in ("gemma-4", "gemma 4")):
            family_match = re.search(r"gemma[- ]4[- ]([a-z0-9]+)", lowered)
            if family_match:
                item.metadata.setdefault("model_family_key", f"gemma-4-{family_match.group(1)}")

    return False


def is_relevant_ai_item(item: IntelligenceItem) -> bool:
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
        "compromise",
        "breach",
        "banned",
        "ban",
        "fear and loathing",
        "drama",
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
    if item.source_type in {"official_news", "model_release", "research", "github_release", "github_repo"}:
        return True
    if item.source_type == "community" and item.metadata.get("recent_feature_signal"):
        return True
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
    order_index: dict[str, int] = {}
    order_counter = 0

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
        family_key = normalized.metadata.get("model_family_key")
        if family_key:
            dedupe_key = f"model-family:{family_key}"
        if title_key and len(title_key) >= 24:
            dedupe_key = title_index.get(title_key, dedupe_key)

        existing = unique_items.get(dedupe_key)

        if _should_replace(existing, normalized):
            unique_items[dedupe_key] = normalized
            if dedupe_key not in order_index:
                order_index[dedupe_key] = order_counter
                order_counter += 1
            if title_key and len(title_key) >= 24:
                title_index[title_key] = dedupe_key

    def _published_timestamp(item: IntelligenceItem) -> float:
        parsed = parse_published_at(item.published_at)
        if parsed is None:
            return float("-inf")
        return parsed.timestamp()

    def _provider_priority(item: IntelligenceItem) -> int:
        haystack = " ".join(
            [
                normalize_text(item.title),
                normalize_text(item.desc),
                normalize_text(item.source_name),
                canonicalize_url(item.url),
            ]
        ).lower()
        return 0 if any(term in haystack for term in _CORE_PROVIDER_TERMS) else 1

    ranked = list(unique_items.values())
    reverse_order_index = {id(item): key for key, item in unique_items.items()}
    ranked.sort(
        key=lambda item: (
            item.priority,
            _provider_priority(item),
            -source_quality_score(item),
            -_published_timestamp(item),
            order_index.get(reverse_order_index.get(id(item), ""), 10**9),
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
