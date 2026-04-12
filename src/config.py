import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in dry-run environments
    def load_dotenv(*_args, **_kwargs) -> None:
        return None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local developer overrides should win over generic .env defaults.
load_dotenv(PROJECT_ROOT / ".env.local")
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")

    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    NOTION_TOKEN = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_INTEGRATION_SECRET")
    NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID") or os.getenv("NOTION_DATABASE_ID")

    DRY_RUN = _get_bool("DRY_RUN", True)
    ENABLE_AI_ANALYSIS = _get_bool("ENABLE_AI_ANALYSIS", False)
    ENABLE_DISCORD_DELIVERY = _get_bool("ENABLE_DISCORD_DELIVERY", False)
    ENABLE_NOTION_DELIVERY = _get_bool("ENABLE_NOTION_DELIVERY", False)
    USE_FIXTURE_DATA = _get_bool("USE_FIXTURE_DATA", False)
    ENABLE_HISTORY_DEDUP = _get_bool("ENABLE_HISTORY_DEDUP", False)
    WRITE_ARTIFACTS = _get_bool("WRITE_ARTIFACTS", True)

    MAX_DESC_LENGTH = int(os.getenv("MAX_DESC_LENGTH", "220"))
    INTERVAL_MINUTES = int(os.getenv("INTERVAL_MINUTES", "15"))
    STOCK_NEWS_LOOKBACK_DAYS = int(os.getenv("STOCK_NEWS_LOOKBACK_DAYS", "7"))
    AI_NEWS_LOOKBACK_DAYS = int(os.getenv("AI_NEWS_LOOKBACK_DAYS", "14"))
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "2000"))
    HISTORY_TTL_HOURS = int(os.getenv("HISTORY_TTL_HOURS", "12"))
    STATE_FILE = os.getenv("STATE_FILE", "data/run_state.json")
    ARTIFACT_FILE = os.getenv("ARTIFACT_FILE", "data/latest_run.json")

    US_STOCKS = _get_list("US_STOCKS", ["NVDA", "TSLA", "AMD", "GOOG", "AAPL"])
    TW_STOCKS = _get_list("TW_STOCKS", ["0050", "2330", "00692"])
    TW_STOCK_SOURCE_ORDER = _get_list("TW_STOCK_SOURCE_ORDER", ["yfinance", "mis"])
    AI_GITHUB_RELEASE_REPOS = _get_list(
        "AI_GITHUB_RELEASE_REPOS",
        [
            "openai/openai-python",
            "anthropics/anthropic-sdk-python",
            "microsoft/autogen",
            "langchain-ai/langchain",
            "crewAIInc/crewAI",
            "huggingface/transformers",
        ],
    )
