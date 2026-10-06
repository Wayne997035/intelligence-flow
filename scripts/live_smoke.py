"""Live smoke test for keyless sources; run in CI where the network is open.

Hits real endpoints (no secrets needed), prints what each collector returned,
then runs the full pipeline as a dry run without AI and prints the digest.
Exits non-zero only on hard failures that mean a collector is broken.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("ENABLE_AI_ANALYSIS", "false")
os.environ.setdefault("ENABLE_HISTORY_DEDUP", "false")

import main  # noqa: E402
from src.collectors.google_news_collector import GoogleNewsCollector  # noqa: E402
from src.collectors.official_ai_collector import OfficialAICollector  # noqa: E402
from src.collectors.stock_collector import StockCollector  # noqa: E402
from src.pipeline import is_relevant_ai_item, normalize_item  # noqa: E402

failures: list[str] = []


def show(label: str, items: list[dict], limit: int = 8) -> None:
    print(f"\n### {label}: {len(items)}")
    for item in items[:limit]:
        print(f"  - {item.get('published_at')} | {item.get('source_name')} | {item.get('title')}")


stocks = StockCollector()
quotes = stocks.fetch_us_stocks()
print("### US quotes")
for quote in quotes:
    print("  ", quote)
if not any(quote["symbol"] == "SPCX" for quote in quotes):
    failures.append("SPCX quote missing")
if quotes and not all("change_pct" in quote for quote in quotes):
    failures.append("quotes missing change_pct")

ticker_news = stocks.fetch_ticker_news()
show("Yahoo ticker news", ticker_news)
show("  of which SPCX", [item for item in ticker_news if "SPCX" in item.get("tags", [])])

google = GoogleNewsCollector()
stock_topics = google.fetch_stock_topics()
ai_topics = google.fetch_ai_topics()
show("Google News stock topics", stock_topics)
show("Google News AI topics", ai_topics, limit=15)
if not stock_topics or not ai_topics:
    failures.append("Google News returned nothing")

official = OfficialAICollector()
feed_items = official._fetch_feed_updates(4)
html_items = official._fetch_html_updates(4)
show("Official RSS feeds", feed_items, limit=30)
show("Official HTML listings", html_items, limit=20)
if not feed_items:
    failures.append("all official RSS feeds empty")

dropped = [
    item["title"]
    for item in ai_topics + feed_items + html_items
    if not is_relevant_ai_item(normalize_item(item))
]
print(f"\n### relevance-dropped among live AI items: {len(dropped)}")
for title in dropped:
    print("  x", title)

print("\n### full pipeline (dry run, no AI)")
result = main.run_job(use_fixture=False, fixture_path=None, enable_ai=False, dry_run=True)
print(result["meta"])

if failures:
    print("\nSMOKE FAILURES:", failures)
    sys.exit(1)
print("\nSMOKE OK")
