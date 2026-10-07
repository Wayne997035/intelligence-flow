"""Scrub account identifiers and credentials from error text.

Provider and HTTP errors embed things like Groq organization ids, NVIDIA
account ids, API keys and full request URLs (which can carry a Discord
webhook token or an `apiKey=` query parameter). GitHub masks the values of
configured secrets in Actions logs, but not identifiers that are not
secrets, and not text written to artifacts, Notion or Discord. Every error
string that leaves the process goes through `redact()` first.
"""
from __future__ import annotations

import re

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # URLs: keep scheme + host, drop path and query (tokens live there).
    (re.compile(r"(https?://[^/\s'\"]+)[^\s'\"]*"), r"\1/…"),
    (re.compile(r"\borg_[A-Za-z0-9]{6,}"), "org_***"),
    (re.compile(r"(account\s*['\"]?)[A-Za-z0-9_\-]{8,}", re.IGNORECASE), r"\1***"),
    (re.compile(r"(Function\s*['\"]?)[0-9a-f\-]{16,}", re.IGNORECASE), r"\1***"),
    (re.compile(r"\b(?:AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{16,}|nvapi-[0-9A-Za-z_\-]{16,}|sk-[0-9A-Za-z_\-]{16,}|ghp_[0-9A-Za-z]{20,}|github_pat_[0-9A-Za-z_]{20,}|secret_[0-9A-Za-z]{20,}|ntn_[0-9A-Za-z]{20,})"), "***"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"((?:api[_-]?key|token|key|secret|password)\s*[=:]\s*)[^\s&'\",;]+", re.IGNORECASE), r"\1***"),
)


def redact(text: object) -> str:
    value = str(text)
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return value
