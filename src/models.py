from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IntelligenceItem:
    title: str
    url: str
    desc: str = ""
    source_name: str = "unknown"
    source_type: str = "unknown"
    published_at: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 999


@dataclass(slots=True)
class ReportItem:
    title: str
    url: str
    summary: str
    insight: str
    source_name: str = "unknown"
    source_type: str = "unknown"
    published_at: str | None = None


@dataclass(slots=True)
class AnalyzedReport:
    title: str
    summary: str
    items: list[ReportItem]
    outlook: str
    outlook_label: str
    metadata: dict[str, Any] = field(default_factory=dict)
