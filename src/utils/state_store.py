from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from pathlib import Path

from src.models import IntelligenceItem
from src.pipeline import canonicalize_url, normalize_text
from src.utils.logger import logger


class RunStateStore:
    def __init__(
        self,
        path: str,
        *,
        enabled: bool = True,
        history_limit: int = 2000,
        ttl_hours: int = 12,
        now_fn=None,
    ):
        self.path = Path(path)
        self.enabled = enabled
        self.history_limit = history_limit
        self.ttl_hours = ttl_hours
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.state = self._load()

    def filter_new_items(
        self,
        category: str,
        items: list[IntelligenceItem],
        *,
        limit: int,
    ) -> tuple[list[IntelligenceItem], int]:
        if not self.enabled:
            return items[:limit], 0

        history = self._get_active_history(category)
        seen = {entry["fingerprint"] for entry in history}
        fresh: list[IntelligenceItem] = []
        skipped = 0
        now_iso = self._now_iso()
        for item in items:
            fingerprint = self._fingerprint(item)
            if fingerprint in seen:
                skipped += 1
                continue
            fresh.append(item)
            seen.add(fingerprint)
            history.append({"fingerprint": fingerprint, "seen_at": now_iso})
            if len(fresh) >= limit:
                break

        self.state[category] = history[-self.history_limit :]
        return fresh, skipped

    def remember(self, category: str, items: list[IntelligenceItem]) -> None:
        if not self.enabled:
            return

        history = self._get_active_history(category)
        seen = {entry["fingerprint"] for entry in history}
        now_iso = self._now_iso()
        for item in items:
            fingerprint = self._fingerprint(item)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            history.append({"fingerprint": fingerprint, "seen_at": now_iso})
        self.state[category] = history[-self.history_limit :]

    def save(self) -> None:
        if not self.enabled:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, ensure_ascii=False, indent=2, sort_keys=True)
        logger.info("Saved run state to %s.", self.path)

    def _load(self) -> dict[str, list[dict[str, str]]]:
        if not self.enabled or not self.path.exists():
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load state store %s: %s", self.path, exc)
            return {}

        normalized: dict[str, list[dict[str, str]]] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, list):
                continue
            entries: list[dict[str, str]] = []
            for entry in value:
                normalized_entry = self._normalize_entry(entry)
                if normalized_entry:
                    entries.append(normalized_entry)
            if entries:
                normalized[key] = entries[-self.history_limit :]
        return normalized

    def _get_active_history(self, category: str) -> list[dict[str, str]]:
        now = self.now_fn()
        active_entries: list[dict[str, str]] = []
        for entry in self.state.get(category, []):
            seen_at = self._parse_seen_at(entry.get("seen_at"))
            if seen_at is None or now - seen_at > timedelta(hours=self.ttl_hours):
                continue
            active_entries.append(entry)
        self.state[category] = active_entries[-self.history_limit :]
        return list(self.state[category])

    def _normalize_entry(self, entry: object) -> dict[str, str] | None:
        if isinstance(entry, str):
            return {"fingerprint": entry, "seen_at": self._now_iso()}
        if not isinstance(entry, dict):
            return None
        fingerprint = entry.get("fingerprint")
        seen_at = entry.get("seen_at")
        if not isinstance(fingerprint, str) or not fingerprint:
            return None
        if not isinstance(seen_at, str) or self._parse_seen_at(seen_at) is None:
            seen_at = self._now_iso()
        return {"fingerprint": fingerprint, "seen_at": seen_at}

    def _parse_seen_at(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _now_iso(self) -> str:
        return self.now_fn().astimezone(timezone.utc).isoformat()

    def _fingerprint(self, item: IntelligenceItem) -> str:
        payload = {
            "title": normalize_text(item.title).lower(),
            "url": canonicalize_url(item.url),
            "source_name": normalize_text(item.source_name).lower(),
            "source_type": normalize_text(item.source_type).lower(),
        }
        return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def dump_artifact(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("Wrote artifact to %s.", target)
