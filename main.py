from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.ai.analyzer import AIAnalyzer
from src.collectors.arxiv_collector import ArxivCollector
from src.collectors.github_release_collector import GitHubReleaseCollector
from src.collectors.hf_collector import HFCollector
from src.collectors.news_collector import NewsCollector
from src.collectors.official_ai_collector import OfficialAICollector
from src.collectors.stock_collector import StockCollector
from src.collectors.tech_collector import TechCollector
from src.config import Config
from src.deliverers.discord_sender import DiscordSender
from src.deliverers.notion_sender import NotionSender
from src.pipeline import (
    ai_impact_score,
    deduplicate_and_rank,
    filter_recent_items,
    is_relevant_ai_item,
    normalize_item,
    parse_published_at,
)
from src.utils.logger import logger
from src.utils.state_store import RunStateStore, dump_artifact

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
except ImportError:  # pragma: no cover - optional in test environments
    BlockingScheduler = None

DEFAULT_FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "sample_bundle.json"


def trim_descriptions(items: list[dict], max_length: int) -> list[dict]:
    trimmed: list[dict] = []
    for item in items:
        new_item = dict(item)
        description = item.get("desc", "")
        if len(description) > max_length:
            new_item["desc"] = description[:max_length].rstrip() + "..."
        trimmed.append(new_item)
    return trimmed


def select_ai_report_candidates(items: list, limit: int) -> list:
    quotas = [
        ("official_news", 6),
        ("news", 4),
        ("model_release", 2),
        ("community", 3),
        ("github_release", 1),
        ("research", 1),
        ("github_repo", 1),
    ]
    selected: list = []
    selected_urls: set[str] = set()

    for source_type, quota in quotas:
        picked = 0
        for item in items:
            if item.source_type != source_type or item.url in selected_urls:
                continue
            selected.append(item)
            selected_urls.add(item.url)
            picked += 1
            if picked >= quota or len(selected) >= limit:
                break
        if len(selected) >= limit:
            return selected[:limit]

    for item in items:
        if item.url in selected_urls:
            continue
        selected.append(item)
        selected_urls.add(item.url)
        if len(selected) >= limit:
            break

    # Guardrail: keep fresh high-signal source types from being squeezed out by quotas.
    required_source_types = ("official_news", "model_release", "github_release")
    for source_type in required_source_types:
        if any(picked.source_type == source_type for picked in selected):
            continue
        candidate = next((item for item in items if item.source_type == source_type), None)
        if candidate and candidate.url not in selected_urls:
            selected.insert(0, candidate)
            selected_urls.add(candidate.url)

    deduped: list = []
    seen_urls: set[str] = set()
    for item in selected:
        if item.url in seen_urls:
            continue
        seen_urls.add(item.url)
        deduped.append(item)

    return deduped[:limit]


def attach_report_appendix(report, selected_items: list, *, summarize_item=None) -> None:
    report_item_urls = {item.url for item in report.items if item.url}
    report_item_titles = {item.title.strip().lower() for item in report.items if item.title}
    appendix_items: list[dict] = []

    for item in selected_items:
        source_type = getattr(item, "source_type", "") or ""
        metadata = getattr(item, "metadata", {}) or {}
        if source_type == "community" and not (metadata.get("recent_feature_signal") or metadata.get("keyword_match")):
            continue

        item_title = (item.title or "").strip().lower()
        if item.url in report_item_urls or (item_title and item_title in report_item_titles):
            continue
        if summarize_item:
            appendix_items.append(summarize_item(item))
        else:
            appendix_items.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "summary": getattr(item, "desc", "") or getattr(item, "summary", ""),
                    "insight": "",
                    "source_name": item.source_name,
                    "source_type": item.source_type,
                    "published_at": item.published_at,
                }
            )

    if appendix_items:
        report.metadata["appendix_items"] = appendix_items


def attach_ai_appendix(report, selected_items: list, *, summarize_item=None) -> None:
    attach_report_appendix(report, selected_items, summarize_item=summarize_item)


def merge_unique_items(*item_groups: list) -> list:
    merged: list = []
    seen_urls: set[str] = set()
    for items in item_groups:
        for item in items:
            url = getattr(item, "url", None) or (item.get("url") if isinstance(item, dict) else "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(item)
    return merged


def load_fixture_bundle(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_bundle_now(bundle: dict) -> datetime | None:
    published_times: list[datetime] = []
    for category in ("stock_news", "ai_news"):
        for item in bundle.get(category, []):
            if not isinstance(item, dict):
                continue
            published_at = item.get("published_at")
            parsed = parse_published_at(published_at)
            if parsed is not None:
                published_times.append(parsed)
    if not published_times:
        return None
    return max(published_times)


def collect_inputs(use_fixture: bool, fixture_path: Path | None = None) -> dict:
    if use_fixture:
        bundle = load_fixture_bundle(fixture_path or DEFAULT_FIXTURE)
        bundle["_fixture"] = True
        logger.info("Loaded fixture bundle from %s.", fixture_path or DEFAULT_FIXTURE)
        return bundle

    stock_fetcher = StockCollector()
    news_fetcher = NewsCollector()
    tech_fetcher = TechCollector()
    hf_fetcher = HFCollector()
    arxiv_fetcher = ArxivCollector()
    official_fetcher = OfficialAICollector()
    github_release_fetcher = GitHubReleaseCollector()
    return {
        "us_stocks": stock_fetcher.fetch_us_stocks(),
        "tw_stocks": stock_fetcher.fetch_tw_stocks(),
        "stock_news": news_fetcher.fetch_stock_news(),
        "ai_news": (
            news_fetcher.fetch_ai_tech_news()
            + official_fetcher.fetch_updates()
            + github_release_fetcher.fetch_latest_releases()
            + tech_fetcher.fetch_all_community_ai()
            + hf_fetcher.fetch_all_hf()
            + arxiv_fetcher.fetch_all_arxiv()
        ),
    }


def build_reports(inputs: dict, *, enable_ai: bool, dry_run: bool, now: datetime | None = None) -> dict:
    analyzer = AIAnalyzer(enable_ai=enable_ai)
    notion = NotionSender(dry_run=dry_run)
    discord = DiscordSender(dry_run=dry_run)
    state_store = RunStateStore(
        Config.STATE_FILE,
        enabled=Config.ENABLE_HISTORY_DEDUP and not inputs.get("_fixture", False) and not dry_run,
        history_limit=Config.HISTORY_LIMIT,
        ttl_hours=Config.HISTORY_TTL_HOURS,
    )

    stock_priority = Config.US_STOCKS + Config.TW_STOCKS
    ai_priority = [
        "Claude",
        "Mythos",
        "Glasswing",
        "Gemini",
        "Anthropic",
        "OpenAI",
        "GPT",
        "ChatGPT",
        "Codex",
        "xAI",
        "Grok",
        "Google",
        "model release",
        "agent",
        "agentic",
        "workflow",
        "tool use",
        "zero-day",
        "vulnerability",
        "cybersecurity",
        "breach",
        "DeepSeek",
        "Llama",
        "Qwen",
        "Mistral",
        "GitHub",
        "arXiv",
        "managed",
    ]

    stock_news_ranked = deduplicate_and_rank(
        trim_descriptions(inputs.get("stock_news", []), Config.MAX_DESC_LENGTH),
        stock_priority,
        limit=48,
        default_source_name="news",
        default_source_type="news",
    )
    ai_raw_trimmed = trim_descriptions(inputs.get("ai_news", []), Config.MAX_DESC_LENGTH)
    ai_input_items: list = []
    ai_irrelevant_count = 0
    for item in ai_raw_trimmed:
        normalized = normalize_item(item)
        if is_relevant_ai_item(normalized):
            ai_input_items.append(item)
        else:
            ai_irrelevant_count += 1
    ai_news_ranked = deduplicate_and_rank(
        ai_input_items,
        ai_priority,
        limit=60,
        default_source_name="unknown",
        default_source_type="news",
    )

    stock_news_recent = filter_recent_items(
        stock_news_ranked,
        max_age_days=Config.STOCK_NEWS_LOOKBACK_DAYS,
        now=now,
        require_published_at=True,
    )
    ai_news_recent = filter_recent_items(
        ai_news_ranked,
        max_age_days=Config.AI_NEWS_LOOKBACK_DAYS,
        now=now,
        require_published_at=True,
    )
    ai_high_impact_archive = filter_recent_items(
        [item for item in ai_news_ranked if ai_impact_score(item) > 0],
        max_age_days=Config.AI_HIGH_IMPACT_LOOKBACK_DAYS,
        now=now,
        require_published_at=True,
    )
    ai_recent_urls = {item.url for item in ai_news_recent}
    ai_recent_cutoff_dropped = {"undated": 0, "older_than_window": 0}
    for item in ai_news_ranked:
        parsed = parse_published_at(item.published_at)
        if parsed is None:
            ai_recent_cutoff_dropped["undated"] += 1
            continue
        if item.url not in ai_recent_urls:
            ai_recent_cutoff_dropped["older_than_window"] += 1

    stock_news, skipped_stock_duplicates = state_store.filter_new_items("stock_news", stock_news_recent, limit=12)
    ai_news, skipped_ai_duplicates = state_store.filter_new_items("ai_news", ai_news_recent, limit=30)

    if not stock_news and stock_news_recent:
        stock_news = stock_news_recent[:12]
    if not ai_news and ai_news_recent:
        ai_news = ai_news_recent[:30]
    ai_news = select_ai_report_candidates(ai_news, limit=24)
    ai_selected_urls = {item.url for item in ai_news}
    ai_recent_not_selected = [item.title for item in ai_news_recent if item.url not in ai_selected_urls][:12]

    stock_report = analyzer.analyze_stock_market(inputs.get("us_stocks", []) + inputs.get("tw_stocks", []), stock_news)
    attach_report_appendix(stock_report, stock_news_recent, summarize_item=analyzer.build_stock_brief_item)
    stock_report.metadata["history_duplicates_skipped"] = skipped_stock_duplicates
    stock_notion_url = notion.create_stock_insight_report(stock_report)
    stock_payload = discord.send_stock_and_analysis(
        inputs.get("us_stocks", []),
        inputs.get("tw_stocks", []),
        stock_report,
        stock_notion_url,
    )

    ai_report = analyzer.analyze_ai_tech(ai_news)
    ai_appendix_pool = merge_unique_items(ai_news_recent, ai_high_impact_archive)
    attach_report_appendix(ai_report, ai_appendix_pool, summarize_item=analyzer.build_ai_brief_item)
    ai_report.metadata["history_duplicates_skipped"] = skipped_ai_duplicates
    ai_notion_url = notion.create_ai_tech_report(ai_report)
    ai_payload = discord.send_ai_tech_report(ai_report, ai_notion_url)

    state_store.remember("stock_news", stock_news)
    state_store.remember("ai_news", ai_news)
    state_store.save()

    result = {
        "stock_report": stock_report,
        "stock_payload": stock_payload,
        "ai_report": ai_report,
        "ai_payload": ai_payload,
        "stock_items": [asdict(normalize_item(item)) for item in stock_news],
        "ai_items": [asdict(normalize_item(item)) for item in ai_news],
        "meta": {
            "dry_run": dry_run,
            "enable_ai": enable_ai,
            "stock_duplicates_skipped": skipped_stock_duplicates,
            "ai_duplicates_skipped": skipped_ai_duplicates,
            "ai_pipeline": {
                "raw_count": len(inputs.get("ai_news", [])),
                "trimmed_count": len(ai_raw_trimmed),
                "irrelevant_dropped": ai_irrelevant_count,
                "ranked_count": len(ai_news_ranked),
                "recent_count": len(ai_news_recent),
                "high_impact_archive_count": len(ai_high_impact_archive),
                "recent_cutoff_dropped": ai_recent_cutoff_dropped,
                "history_or_limit_count": len(ai_news),
                "recent_not_selected_sample": ai_recent_not_selected,
            },
        },
    }
    if Config.WRITE_ARTIFACTS:
        dump_artifact(
            Config.ARTIFACT_FILE,
            {
                "stock_report": asdict(stock_report),
                "stock_payload": stock_payload,
                "ai_report": asdict(ai_report),
                "ai_payload": ai_payload,
                "meta": result["meta"],
            },
        )
    return result


def validate_runtime(*, enable_ai: bool, dry_run: bool) -> None:
    if enable_ai:
        if not (Config.GEMINI_API_KEY or Config.GROQ_API_KEY):
            raise RuntimeError("AI analysis requested but neither GEMINI_API_KEY nor GROQ_API_KEY is configured.")

    if not dry_run:
        if Config.ENABLE_DISCORD_DELIVERY and not Config.DISCORD_WEBHOOK_URL:
            raise RuntimeError("Live Discord delivery enabled but DISCORD_WEBHOOK_URL is missing.")
        if Config.ENABLE_NOTION_DELIVERY and not (Config.NOTION_TOKEN and Config.NOTION_PAGE_ID):
            raise RuntimeError("Live Notion delivery enabled but NOTION_TOKEN or NOTION_PAGE_ID is missing.")


def run_job(*, use_fixture: bool, fixture_path: Path | None, enable_ai: bool, dry_run: bool) -> dict:
    logger.info(
        "Intel-Flow cycle started (fixture=%s, enable_ai=%s, dry_run=%s).",
        use_fixture,
        enable_ai,
        dry_run,
    )
    inputs = collect_inputs(use_fixture, fixture_path)
    reference_now = infer_bundle_now(inputs) if use_fixture else None
    return build_reports(inputs, enable_ai=enable_ai, dry_run=dry_run, now=reference_now)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intel-Flow Market Intelligence")
    parser.add_argument("--once", action="store_true", help="執行一次後退出")
    parser.add_argument("--schedule", action="store_true", help="依排程重複執行")
    parser.add_argument("--use-fixture", action="store_true", help="使用本地 fixture，不打外部來源")
    parser.add_argument("--fixture-path", type=Path, help="自訂 fixture JSON 路徑")
    parser.add_argument("--enable-ai", action="store_true", help="允許呼叫 Gemini/Groq 分析")
    parser.add_argument("--live-delivery", action="store_true", help="允許發送到 Discord / Notion")
    return parser.parse_args()


def resolve_runtime_options(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    dry_run = Config.DRY_RUN
    if args.live_delivery:
        dry_run = False

    use_fixture = args.use_fixture or Config.USE_FIXTURE_DATA
    enable_ai = args.enable_ai or Config.ENABLE_AI_ANALYSIS
    return dry_run, use_fixture, enable_ai


if __name__ == "__main__":
    args = parse_args()
    dry_run, use_fixture, enable_ai = resolve_runtime_options(args)
    validate_runtime(enable_ai=enable_ai, dry_run=dry_run)

    if args.schedule:
        if BlockingScheduler is None:
            raise RuntimeError("apscheduler is not installed; schedule mode is unavailable.")
        scheduler = BlockingScheduler()
        scheduler.add_job(
            lambda: run_job(
                use_fixture=use_fixture,
                fixture_path=args.fixture_path,
                enable_ai=enable_ai,
                dry_run=dry_run,
            ),
            "cron",
            hour="8,20",
            minute=0,
            id="intel_flow_job",
        )
        logger.info("Schedule mode enabled for 08:00 and 20:00.")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")
    else:
        run_job(
            use_fixture=use_fixture,
            fixture_path=args.fixture_path,
            enable_ai=enable_ai,
            dry_run=dry_run,
        )
