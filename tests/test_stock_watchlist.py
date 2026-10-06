import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import main
from src.collectors import stock_collector
from src.collectors.google_news_collector import GoogleNewsCollector
from src.collectors.news_collector import stock_news_keywords
from src.collectors.stock_collector import StockCollector
from src.config import Config, _get_alias_map
from src.deliverers.discord_sender import DiscordSender
from src.models import IntelligenceItem


def _history(closes, volumes=None):
    volumes = volumes or [1000] * len(closes)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": volumes,
        }
    )


class TestWatchlistConfig(unittest.TestCase):
    def test_spacex_is_watched_by_default(self):
        self.assertIn("SPCX", Config.US_STOCKS)
        self.assertIn("SpaceX", Config.STOCK_NAME_ALIASES["SPCX"])

    def test_alias_map_env_parsing(self):
        with patch.dict("os.environ", {"X_ALIASES": "SPCX:SpaceX|Starlink, NVDA:Nvidia,bad"}):
            parsed = _get_alias_map("X_ALIASES", {})
        self.assertEqual(parsed, {"SPCX": ["SpaceX", "Starlink"], "NVDA": ["Nvidia"]})

    def test_stock_news_keywords_include_aliases_without_duplicates(self):
        keywords = stock_news_keywords()
        self.assertIn("SPCX", keywords)
        self.assertIn("SpaceX", keywords)
        self.assertIn("Starlink", keywords)
        self.assertEqual(len(keywords), len(set(keywords)))


class TestStockQuoteMetrics(unittest.TestCase):
    def test_quote_from_history_computes_pct_and_trend(self):
        closes = [100.0, 101, 102, 103, 104, 105, 110.0]
        volumes = [1000, 1000, 1000, 1000, 1000, 1000, 3000]
        quote = StockCollector()._quote_from_history("SPCX", _history(closes, volumes), source="yfinance")
        self.assertEqual(quote["price"], 110.0)
        self.assertEqual(quote["change"], "+5.00")
        self.assertEqual(quote["change_pct"], 4.76)
        self.assertEqual(quote["change_5d_pct"], round((110 - 101) / 101 * 100, 2))
        self.assertEqual(quote["change_1m_pct"], 10.0)
        self.assertEqual(quote["volume_ratio"], 3.0)
        self.assertEqual(quote["range_1m"], "99.00-111.00")

    def test_quote_from_empty_history_is_none(self):
        self.assertIsNone(StockCollector()._quote_from_history("X", pd.DataFrame(), source="yfinance"))

    def test_normalize_ticker_news_supports_nested_and_flat_shapes(self):
        collector = StockCollector()
        nested = collector._normalize_ticker_news(
            "SPCX",
            {
                "content": {
                    "title": "SpaceX shares jump after Starship flight",
                    "canonicalUrl": {"url": "https://finance.yahoo.com/news/spacex-1"},
                    "pubDate": "2026-10-05T12:00:00Z",
                    "provider": {"displayName": "Reuters"},
                    "summary": "Shares rose.",
                }
            },
        )
        self.assertEqual(nested["source_name"], "Reuters")
        self.assertEqual(nested["tags"], ["SPCX"])
        flat = collector._normalize_ticker_news(
            "SPCX",
            {"title": "Starlink update", "link": "https://x.test/a", "providerPublishTime": 1790000000, "publisher": "Bloomberg"},
        )
        self.assertEqual(flat["source_name"], "Bloomberg")
        self.assertTrue(flat["published_at"].startswith("2026-"))
        self.assertIsNone(collector._normalize_ticker_news("SPCX", {"content": {"title": "no url"}}))

    def test_fetch_ticker_news_falls_back_to_yahoo_rss(self):
        fake_feed = SimpleNamespace(
            entries=[
                {
                    "title": "SpaceX stock climbs to highest since June",
                    "link": "https://finance.yahoo.com/news/spacex-climbs",
                    "published": "Mon, 05 Oct 2026 20:04:38 +0000",
                    "summary": "Shares rose 7.6%.",
                }
            ]
        )
        with patch.object(stock_collector, "yf", None), patch.object(
            stock_collector, "feedparser"
        ) as mock_feedparser, patch.object(Config, "US_STOCKS", ["SPCX"]):
            mock_feedparser.parse.return_value = fake_feed
            collector = StockCollector()
            items = collector.fetch_ticker_news()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["tags"], ["SPCX"])
        self.assertEqual(items[0]["published_at"], "2026-10-05T20:04:38+00:00")
        self.assertIn("s=SPCX", mock_feedparser.parse.call_args.args[0])

    def test_fetch_ticker_news_disabled(self):
        with patch.object(Config, "ENABLE_TICKER_NEWS", False):
            self.assertEqual(StockCollector().fetch_ticker_news(), [])


class TestGoogleNewsCollector(unittest.TestCase):
    def test_build_url_encodes_topic_and_window(self):
        url = GoogleNewsCollector().build_url("SpaceX Starlink", days_back=7)
        self.assertIn("q=SpaceX+Starlink+when%3A7d", url)

    @patch("src.collectors.google_news_collector.feedparser")
    def test_fetch_topics_strips_publisher_suffix(self, mock_feedparser):
        mock_feedparser.parse.return_value = SimpleNamespace(
            entries=[
                {
                    "title": "SpaceX wins new launch contract - Reuters",
                    "link": "https://news.google.com/articles/abc",
                    "published": "Mon, 05 Oct 2026 08:00:00 GMT",
                    "source": {"title": "Reuters"},
                },
                {"title": "", "link": "https://news.google.com/articles/skip"},
            ]
        )
        with patch.object(Config, "STOCK_WATCH_TOPICS", ["SpaceX"]):
            items = GoogleNewsCollector().fetch_stock_topics()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "SpaceX wins new launch contract")
        self.assertEqual(items[0]["source_name"], "Reuters")
        self.assertEqual(items[0]["published_at"], "2026-10-05T08:00:00+00:00")

    def test_disabled_flag_skips_fetch(self):
        with patch.object(Config, "ENABLE_GOOGLE_NEWS", False):
            self.assertEqual(GoogleNewsCollector().fetch_ai_topics(), [])


class TestBalanceStockNews(unittest.TestCase):
    def _item(self, title):
        return IntelligenceItem(title=title, url=f"https://x.test/{title}", source_type="news")

    def test_round_robin_keeps_later_symbols_in_top_slots(self):
        items = [self._item(f"Nvidia story {i}") for i in range(5)] + [
            self._item("SpaceX Starship launch"),
            self._item("Chip sector outlook"),
        ]
        with patch.object(Config, "US_STOCKS", ["NVDA", "SPCX"]), patch.object(Config, "TW_STOCKS", []):
            balanced = main.balance_stock_news(items)
        self.assertEqual(
            [item.title for item in balanced[:3]],
            ["Nvidia story 0", "SpaceX Starship launch", "Chip sector outlook"],
        )
        self.assertEqual(len(balanced), len(items))

    def test_stock_priority_includes_aliases(self):
        priority = main.build_stock_priority()
        self.assertLess(priority.index("SPCX"), priority.index("SpaceX"))


class TestDiscordQuoteTrend(unittest.TestCase):
    def test_render_quotes_shows_pct_and_trend(self):
        rendered = DiscordSender(dry_run=True)._render_quotes(
            [
                {
                    "symbol": "SPCX",
                    "price": 160.95,
                    "change": "+5.00",
                    "change_pct": 3.21,
                    "range": "150.00-162.00",
                    "change_5d_pct": 4.5,
                    "change_1m_pct": -2.25,
                    "volume_ratio": 1.8,
                }
            ]
        )
        self.assertIn("變:+5.00 (+3.21%)", rendered)
        self.assertIn("5日:+4.5% | 1月:-2.2% | 量比:1.80x", rendered)


if __name__ == "__main__":
    unittest.main()


class TestLaunchStoryDedupe(unittest.TestCase):
    def test_main_report_keeps_one_item_per_model_launch(self):
        items = [
            IntelligenceItem(title=title, url=f"https://x.test/{i}", source_type="news")
            for i, title in enumerate(
                [
                    "Google rolls out Gemini 4 Argon, its most advanced AI model",
                    "Google announces Gemini 4 flagship AI model after months of delays",
                    "Anthropic releases Claude Sonnet 5.5",
                    "Barclays expands use of Claude",
                ]
            )
        ]
        selected = main.select_ai_report_candidates(items, limit=10)
        titles = [item.title for item in selected]
        self.assertEqual(sum("Gemini 4" in title for title in titles), 1)
        self.assertIn("Anthropic releases Claude Sonnet 5.5", titles)
        self.assertIn("Barclays expands use of Claude", titles)
