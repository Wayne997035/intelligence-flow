"""Print the non-secret settings this run uses, so a change to a GitHub
repository variable can be checked in the Actions log."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402

SETTINGS = (
    "US_STOCKS",
    "TW_STOCKS",
    "STOCK_NAME_ALIASES",
    "STOCK_WATCH_TOPICS",
    "TW_STOCK_SOURCE_ORDER",
    "AI_WATCH_TOPICS",
    "AI_MODEL_WATCH",
    "AI_GITHUB_RELEASE_REPOS",
    "STOCK_NEWS_LOOKBACK_DAYS",
    "AI_NEWS_LOOKBACK_DAYS",
    "AI_HIGH_IMPACT_LOOKBACK_DAYS",
    "HISTORY_TTL_HOURS",
    "ENABLE_GOOGLE_NEWS",
    "ENABLE_TICKER_NEWS",
    "AI_PROVIDER_ORDER",
    "AI_MODEL",
    "GROQ_MODEL",
    "NVIDIA_MODEL",
    "NVIDIA_ALLOW_PRODUCTION",
)
KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "NEWS_API_KEY", "DISCORD_WEBHOOK_URL", "NOTION_TOKEN")

for name in SETTINGS:
    print(f"{name} = {getattr(Config, name)!r}")
print("API keys set:", ", ".join(f"{key}={'yes' if getattr(Config, key) else 'no'}" for key in KEYS))
