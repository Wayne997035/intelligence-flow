from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.ai.analyzer import AIAnalyzer
from src.collectors.arxiv_collector import ArxivCollector
from src.collectors.github_release_collector import GitHubReleaseCollector
from src.collectors.google_news_collector import GoogleNewsCollector
from src.collectors.hf_collector import HFCollector
from src.collectors.news_collector import NewsCollector
from src.collectors.official_ai_collector import OfficialAICollector
from src.collectors.stock_collector import StockCollector
from src.collectors.tech_collector import TechCollector
from src.config import Config
from src.deliverers.discord_sender import DiscordSender
from src.deliverers.notion_sender import NotionSender
from src.models import AnalyzedReport
from src.pipeline import (
    ai_impact_score,
    deduplicate_and_rank,
    filter_recent_items,
    is_relevant_ai_item,
    launch_story_key,
    normalize_item,
    parse_published_at,
    source_quality_score,
)
from src.utils.logger import logger
from src.utils.redact import redact
from src.utils.state_store import RunStateStore, dump_artifact

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
except ImportError:  # pragma: no cover - optional in test environments
    BlockingScheduler = None

DEFAULT_FIXTURE = Path(__file__).parent / "tests" / "fixtures" / "sample_bundle.json"


def safe_call(errors: list[str], label: str, fn, default):
    """Run one pipeline step; on any exception log it, record it and return
    `default` so a single broken source, report or delivery never stops the
    rest of the run."""
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - exercised via tests with fakes
        message = f"{label}: {type(exc).__name__}: {redact(exc)[:200]}"
        logger.exception("Pipeline step failed - %s", message)
        errors.append(message)
        return default


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
    """Quota-based pick for the main AI report, at most one item per model
    launch: three outlets covering "Gemini 4 Argon" should take one slot, not
    three. The extra coverage still reaches the Notion appendix."""
    # Keep the story's slot where it first ranks, but fill it with the
    # highest-quality source (the provider's own post over press coverage).
    best_by_story: dict[str, object] = {}
    for item in items:
        story = launch_story_key(f"{item.title} {item.desc}", source_type=item.source_type)
        if story and (
            story not in best_by_story or source_quality_score(item) > source_quality_score(best_by_story[story])
        ):
            best_by_story[story] = item

    placed: set[str] = set()
    unique_story_items: list = []
    for item in items:
        story = launch_story_key(f"{item.title} {item.desc}", source_type=item.source_type)
        if story:
            if story in placed:
                continue
            placed.add(story)
            item = best_by_story[story]
        unique_story_items.append(item)
    return _select_ai_report_candidates(unique_story_items, limit)


def _select_ai_report_candidates(items: list, limit: int) -> list:
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


def build_stock_priority() -> list[str]:
    """Symbols interleaved with their aliases so a "SpaceX" headline ranks like "SPCX"."""
    priority: list[str] = []
    for symbol in Config.US_STOCKS + Config.TW_STOCKS:
        for keyword in (symbol, *Config.STOCK_NAME_ALIASES.get(symbol, [])):
            if keyword not in priority:
                priority.append(keyword)
    return priority


def balance_stock_news(items: list) -> list:
    """Interleave ranked stock news round-robin across watched symbols.

    deduplicate_and_rank orders by the index of the first matched keyword, so
    every NVDA headline would otherwise outrank every SPCX headline and the
    later symbols in the watchlist would never reach the report's slots.
    Items not tied to any symbol (sector themes) form their own trailing group.
    """
    groups: dict[str, list] = {}
    order: list[str] = []
    for item in items:
        text = f"{item.title} {item.desc}".lower()
        group = next(
            (
                symbol
                for symbol in Config.US_STOCKS + Config.TW_STOCKS
                if any(
                    keyword.lower() in text or keyword in (item.tags or [])
                    for keyword in (symbol, *Config.STOCK_NAME_ALIASES.get(symbol, []))
                )
            ),
            "_other",
        )
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(item)

    if "_other" in order:
        order.remove("_other")
        order.append("_other")
    balanced: list = []
    while any(groups[group] for group in order):
        for group in order:
            if groups[group]:
                balanced.append(groups[group].pop(0))
    return balanced


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

    errors: list[str] = []
    collector_classes = {
        "stock": StockCollector,
        "news": NewsCollector,
        "tech": TechCollector,
        "hf": HFCollector,
        "arxiv": ArxivCollector,
        "official": OfficialAICollector,
        "github_release": GitHubReleaseCollector,
        "google_news": GoogleNewsCollector,
    }
    # Built one by one so a collector that fails to construct only loses its
    # own sources.
    fetchers = {
        key: fetcher
        for key, cls in collector_classes.items()
        if (fetcher := safe_call(errors, f"collector setup {key}", cls, None)) is not None
    }

    def fetch(label: str, fetcher_key: str, method: str) -> list:
        fetcher = fetchers.get(fetcher_key)
        if fetcher is None:
            return []
        return safe_call(errors, f"source {label}", getattr(fetcher, method), []) or []

    stock_sources = {
        "newsapi": fetch("stock/newsapi", "news", "fetch_stock_news"),
        "ticker_news": fetch("stock/ticker_news", "stock", "fetch_ticker_news"),
        "google_news": fetch("stock/google_news", "google_news", "fetch_stock_topics"),
    }
    ai_sources = {
        "newsapi": fetch("ai/newsapi", "news", "fetch_ai_tech_news"),
        "google_news": fetch("ai/google_news", "google_news", "fetch_ai_topics"),
        "official": fetch("ai/official", "official", "fetch_updates"),
        "github_release": fetch("ai/github_release", "github_release", "fetch_latest_releases"),
        "community": fetch("ai/community", "tech", "fetch_all_community_ai"),
        "huggingface": fetch("ai/huggingface", "hf", "fetch_all_hf"),
        "arxiv": fetch("ai/arxiv", "arxiv", "fetch_all_arxiv"),
    }
    source_counts = {
        "stock_news": {name: len(items) for name, items in stock_sources.items()},
        "ai_news": {name: len(items) for name, items in ai_sources.items()},
    }
    logger.info("Collected item counts per source: %s", source_counts)
    return {
        "us_stocks": fetch("quotes/us", "stock", "fetch_us_stocks"),
        "tw_stocks": fetch("quotes/tw", "stock", "fetch_tw_stocks"),
        "stock_news": [item for items in stock_sources.values() for item in items],
        "ai_news": [item for items in ai_sources.values() for item in items],
        "_source_counts": source_counts,
        "_errors": errors,
    }


def build_reports(inputs: dict, *, enable_ai: bool, dry_run: bool, now: datetime | None = None) -> dict:
    """Build and deliver both reports.

    Failure isolation: each step runs through safe_call, the stock and AI
    reports are independent sections, Notion and Discord failures are
    recorded by the senders instead of raised, and a section that cannot
    produce its report still sends a short failure notice. All failures land
    in meta["errors"] (and latest_run.json) so the workflow can go red after
    delivery instead of the run dying half way.
    """
    errors: dict[str, list[str]] = {
        "collection": list(inputs.get("_errors", [])),
        "report": [],
        "delivery": [],
        "state": [],
    }

    # Dry runs are evaluation, which the NVIDIA trial terms allow; live
    # delivery only uses NVIDIA when NVIDIA_ALLOW_PRODUCTION is set.
    analyzer = safe_call(
        errors["report"],
        "analyzer setup",
        lambda: AIAnalyzer(enable_ai=enable_ai, nvidia_enabled=Config.NVIDIA_ALLOW_PRODUCTION or dry_run),
        None,
    ) or AIAnalyzer(enable_ai=False)
    notion = safe_call(errors["delivery"], "notion setup", lambda: NotionSender(dry_run=dry_run), None) or NotionSender(
        dry_run=dry_run, enabled=False
    )
    discord = safe_call(
        errors["delivery"], "discord setup", lambda: DiscordSender(dry_run=dry_run), None
    ) or DiscordSender(dry_run=dry_run, enabled=False)
    # `not dry_run` here is unrelated to the delivery-guard choke point in
    # src/deliverers/guard.py.
    #
    # In production it never fires: the scheduled workflow always passes
    # `--live-delivery`, so dry_run is False on every real run and this
    # condition has no say over production history dedup.
    #
    # On a local dev machine it does matter, and that is the only reason it is
    # kept: the filesystem persists between runs, so without it a throwaway
    # dry-run would write throwaway fingerprints into the same on-disk dedup
    # history that later runs read back.
    state_store = safe_call(
        errors["state"],
        "state store setup",
        lambda: RunStateStore(
            Config.STATE_FILE,
            enabled=Config.ENABLE_HISTORY_DEDUP and not inputs.get("_fixture", False) and not dry_run,
            history_limit=Config.HISTORY_LIMIT,
            ttl_hours=Config.HISTORY_TTL_HOURS,
        ),
        None,
    ) or RunStateStore(Config.STATE_FILE, enabled=False)

    quotes = list(inputs.get("us_stocks", [])) + list(inputs.get("tw_stocks", []))
    meta: dict = {
        "dry_run": dry_run,
        "enable_ai": enable_ai,
        "source_counts": inputs.get("_source_counts", {}),
    }

    def stock_section() -> dict:
        stock_news_ranked = deduplicate_and_rank(
            trim_descriptions(inputs.get("stock_news", []), Config.MAX_DESC_LENGTH),
            build_stock_priority(),
            limit=48,
            default_source_name="news",
            default_source_type="news",
        )
        stock_news_recent = balance_stock_news(
            filter_recent_items(
                stock_news_ranked,
                max_age_days=Config.STOCK_NEWS_LOOKBACK_DAYS,
                now=now,
                require_published_at=True,
            )
        )
        stock_news, skipped = safe_call(
            errors["state"],
            "state filter stock_news",
            lambda: state_store.filter_new_items("stock_news", stock_news_recent, limit=12),
            (stock_news_recent[:12], 0),
        )
        if not stock_news and stock_news_recent:
            stock_news = stock_news_recent[:12]

        report = safe_call(
            errors["report"],
            "stock analysis",
            lambda: analyzer.analyze_stock_market(quotes, stock_news),
            None,
        ) or analyzer._build_stock_fallback(quotes, stock_news)
        safe_call(
            errors["report"],
            "stock appendix",
            lambda: attach_report_appendix(report, stock_news_recent, summarize_item=analyzer.build_stock_brief_item),
            None,
        )
        report.metadata["history_duplicates_skipped"] = skipped
        notion_url = notion.create_stock_insight_report(report)
        payload = discord.send_stock_and_analysis(
            inputs.get("us_stocks", []), inputs.get("tw_stocks", []), report, notion_url
        )
        meta["stock_duplicates_skipped"] = skipped
        return {"report": report, "payload": payload, "items": stock_news}

    ai_priority = [
        "Claude",
        "Opus",
        "Sonnet",
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

    def ai_section() -> dict:
        ai_raw_trimmed = trim_descriptions(inputs.get("ai_news", []), Config.MAX_DESC_LENGTH)
        ai_input_items: list = []
        ai_irrelevant_count = 0
        ai_irrelevant_sample: list[str] = []
        for item in ai_raw_trimmed:
            normalized = normalize_item(item)
            if is_relevant_ai_item(normalized):
                ai_input_items.append(item)
            else:
                ai_irrelevant_count += 1
                if len(ai_irrelevant_sample) < 12:
                    ai_irrelevant_sample.append(normalized.title)
        ai_news_ranked = deduplicate_and_rank(
            ai_input_items,
            ai_priority,
            limit=60,
            default_source_name="unknown",
            default_source_type="news",
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

        ai_news, skipped = safe_call(
            errors["state"],
            "state filter ai_news",
            lambda: state_store.filter_new_items("ai_news", ai_news_recent, limit=30),
            (ai_news_recent[:30], 0),
        )
        if not ai_news and ai_news_recent:
            ai_news = ai_news_recent[:30]
        ai_news = select_ai_report_candidates(ai_news, limit=24)
        ai_selected_urls = {item.url for item in ai_news}

        report = safe_call(
            errors["report"],
            "ai analysis",
            lambda: analyzer.analyze_ai_tech(ai_news),
            None,
        ) or analyzer._build_ai_fallback(ai_news)
        safe_call(
            errors["report"],
            "ai appendix",
            lambda: attach_report_appendix(
                report,
                merge_unique_items(ai_news_recent, ai_high_impact_archive),
                summarize_item=analyzer.build_ai_brief_item,
            ),
            None,
        )
        report.metadata["history_duplicates_skipped"] = skipped
        notion_url = notion.create_ai_tech_report(report)
        payload = discord.send_ai_tech_report(report, notion_url)
        meta["ai_duplicates_skipped"] = skipped
        meta["ai_pipeline"] = {
            "raw_count": len(inputs.get("ai_news", [])),
            "trimmed_count": len(ai_raw_trimmed),
            "irrelevant_dropped": ai_irrelevant_count,
            "irrelevant_dropped_sample": ai_irrelevant_sample,
            "ranked_count": len(ai_news_ranked),
            "recent_count": len(ai_news_recent),
            "high_impact_archive_count": len(ai_high_impact_archive),
            "recent_cutoff_dropped": ai_recent_cutoff_dropped,
            "history_or_limit_count": len(ai_news),
            "recent_not_selected_sample": [item.title for item in ai_news_recent if item.url not in ai_selected_urls][
                :12
            ],
        }
        return {"report": report, "payload": payload, "items": ai_news}

    def failed_section(title: str, label: str) -> dict:
        reason = next((error for error in reversed(errors["report"]) if error.startswith(label)), "未知錯誤")
        report = AnalyzedReport(
            title=title,
            summary=f"⚠️ 本輪{title}產生失敗（{reason}），請查看 GitHub Actions log。",
            items=[],
            outlook="",
            outlook_label="",
            metadata={"mode": "failed", "section_error": reason},
        )
        payload = safe_call(
            errors["delivery"],
            f"{label} failure notice",
            lambda: discord.send_report_failure_notice(title, reason),
            {},
        )
        return {"report": report, "payload": payload, "items": []}

    stock = safe_call(errors["report"], "stock section", stock_section, None) or failed_section(
        "投資情報報告", "stock section"
    )
    ai = safe_call(errors["report"], "ai section", ai_section, None) or failed_section("AI 技術前沿情報", "ai section")
    stock_report, ai_report = stock["report"], ai["report"]

    safe_call(errors["state"], "state remember stock_news", lambda: state_store.remember("stock_news", stock["items"]), None)
    safe_call(errors["state"], "state remember ai_news", lambda: state_store.remember("ai_news", ai["items"]), None)
    safe_call(errors["state"], "state save", state_store.save, None)

    errors["delivery"].extend(notion.errors + discord.errors)
    meta.setdefault("stock_duplicates_skipped", 0)
    meta.setdefault("ai_duplicates_skipped", 0)
    meta["ai_errors"] = {
        name: report.metadata["ai_error"]
        for name, report in (("stock", stock_report), ("ai", ai_report))
        if report.metadata.get("ai_error")
    }
    meta["errors"] = {kind: messages for kind, messages in errors.items() if messages}

    result = {
        "stock_report": stock_report,
        "stock_payload": stock["payload"],
        "ai_report": ai_report,
        "ai_payload": ai["payload"],
        "stock_items": [asdict(normalize_item(item)) for item in stock["items"]],
        "ai_items": [asdict(normalize_item(item)) for item in ai["items"]],
        "meta": meta,
    }
    if Config.WRITE_ARTIFACTS:
        safe_call(
            errors["state"],
            "write artifact",
            lambda: dump_artifact(
                Config.ARTIFACT_FILE,
                {
                    "stock_report": asdict(stock_report),
                    "stock_payload": stock["payload"],
                    "ai_report": asdict(ai_report),
                    "ai_payload": ai["payload"],
                    "meta": meta,
                },
            ),
            None,
        )
    if meta["errors"]:
        logger.error("Run finished with errors: %s", meta["errors"])
    return result


def validate_runtime(*, enable_ai: bool, dry_run: bool) -> None:
    if enable_ai:
        if not (Config.GEMINI_API_KEY or Config.GROQ_API_KEY or Config.NVIDIA_API_KEY):
            raise RuntimeError(
                "AI analysis requested but none of GEMINI_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY is configured."
            )

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
            hour="9,18",
            minute=0,
            id="intel_flow_job",
        )
        logger.info("Schedule mode enabled for 09:00 and 18:00.")
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
