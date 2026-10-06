import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in dry-run environments
    def load_dotenv(*_args, **_kwargs) -> None:
        return None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Default local config source. Can be overridden by ENV_FILE.
ENV_FILE = os.getenv("ENV_FILE", ".env.local")
load_dotenv(PROJECT_ROOT / ENV_FILE)
if ENV_FILE != ".env":
    # Load `.env` as a fallback for non-secret toggles (does not override existing vars).
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


def _get_alias_map(name: str, default: dict[str, list[str]]) -> dict[str, list[str]]:
    """Parse `SYMBOL:Alias A|Alias B,SYMBOL2:Alias` into {symbol: [aliases]}."""
    value = os.getenv(name)
    if not value:
        return default
    parsed: dict[str, list[str]] = {}
    for entry in value.split(","):
        symbol, _, raw_aliases = entry.partition(":")
        symbol = symbol.strip()
        aliases = [alias.strip() for alias in raw_aliases.split("|") if alias.strip()]
        if symbol and aliases:
            parsed[symbol] = aliases
    return parsed or default


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip()
    if not cleaned:
        return default
    if re.fullmatch(r"\{[A-Z0-9_]+\}", cleaned):
        return default
    return cleaned


class Config:
    GEMINI_API_KEY = _get_env("GEMINI_API_KEY")
    GROQ_API_KEY = _get_env("GROQ_API_KEY")
    AI_MODEL = _get_env("AI_MODEL", "gemini-flash-latest")
    GROQ_MODEL = _get_env("GROQ_MODEL", "llama-3.3-70b-versatile")

    NEWS_API_KEY = _get_env("NEWS_API_KEY")
    GITHUB_TOKEN = _get_env("GITHUB_TOKEN")

    DISCORD_WEBHOOK_URL = _get_env("DISCORD_WEBHOOK_URL")
    NOTION_TOKEN = _get_env("NOTION_TOKEN") or _get_env("NOTION_INTEGRATION_SECRET")
    NOTION_PAGE_ID = _get_env("NOTION_PAGE_ID") or _get_env("NOTION_DATABASE_ID")
    # Dedicated database for cross-run dedup state persistence (separate from
    # NOTION_PAGE_ID, which is the report delivery destination).
    NOTION_STATE_DB_ID = _get_env("NOTION_STATE_DB_ID")
    NOTION_STATE_DS_ID = _get_env("NOTION_STATE_DS_ID")

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
    AI_NEWS_LOOKBACK_DAYS = int(os.getenv("AI_NEWS_LOOKBACK_DAYS", "7"))
    AI_HIGH_IMPACT_LOOKBACK_DAYS = int(os.getenv("AI_HIGH_IMPACT_LOOKBACK_DAYS", "30"))
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "2000"))
    # 26h covers both schedule gaps (10:30->20:00 = 9.5h, 20:00->next 10:30 =
    # 14.5h) with buffer for a delayed run; 24h alone would let the
    # overnight gap's dedup window lapse right at the boundary.
    HISTORY_TTL_HOURS = int(os.getenv("HISTORY_TTL_HOURS", "26"))
    STATE_FILE = _get_env("STATE_FILE", "data/run_state.json") or "data/run_state.json"
    ARTIFACT_FILE = _get_env("ARTIFACT_FILE", "data/latest_run.json") or "data/latest_run.json"

    US_STOCKS = _get_list("US_STOCKS", ["NVDA", "TSLA", "AMD", "GOOG", "AAPL", "SPCX"])
    TW_STOCKS = _get_list("TW_STOCKS", ["0050", "2330", "00692"])
    # Ticker symbols alone are poor news search terms ("SPCX" rarely appears in
    # headlines, "SpaceX"/"Starlink" do), so each watched symbol maps to the
    # company/product names used for news queries and ranking.
    STOCK_NAME_ALIASES = _get_alias_map(
        "STOCK_NAME_ALIASES",
        {
            "NVDA": ["Nvidia"],
            "TSLA": ["Tesla"],
            "AMD": ["AMD"],
            "GOOG": ["Alphabet", "Google"],
            "AAPL": ["Apple"],
            "SPCX": ["SpaceX", "Starlink", "Starship"],
            "2330": ["TSMC"],
        },
    )
    # Extra topics tracked on the stock side via Google News RSS (no API key).
    STOCK_WATCH_TOPICS = _get_list(
        "STOCK_WATCH_TOPICS",
        ["SpaceX stock", "SpaceX Starlink", "SpaceX Starship launch", "Nvidia earnings", "TSMC"],
    )
    AI_WATCH_TOPICS = _get_list(
        "AI_WATCH_TOPICS",
        [
            "Anthropic Claude",
            "Claude Opus release",
            "OpenAI GPT-6",
            "OpenAI new model",
            "Google Gemini 4",
            "xAI Grok",
            "Meta Llama",
            "DeepSeek",
            "AI agent launch",
        ],
    )
    # Current flagship model names used as exact-phrase news queries. Version
    # numbers go stale quickly, so keep this list editable via env instead of
    # hardcoding it in the collector.
    AI_MODEL_WATCH = _get_list(
        "AI_MODEL_WATCH",
        [
            "Claude Opus",
            "Claude Sonnet",
            "Claude Fable",
            "Claude Mythos",
            "GPT-6",
            "GPT-6 Astra",
            "GPT-6.1 Sol",
            "Gemini 4",
            "Grok",
        ],
    )
    ENABLE_GOOGLE_NEWS = _get_bool("ENABLE_GOOGLE_NEWS", True)
    ENABLE_TICKER_NEWS = _get_bool("ENABLE_TICKER_NEWS", True)
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
