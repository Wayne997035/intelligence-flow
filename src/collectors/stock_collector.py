from __future__ import annotations

from datetime import datetime, timezone

import requests

from src.config import Config
from src.utils.logger import logger

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency in dry-run
    yf = None


class StockCollector:
    def __init__(self):
        self.us_tickers = Config.US_STOCKS
        self.tw_tickers = Config.TW_STOCKS
        self.tw_source_order = Config.TW_STOCK_SOURCE_ORDER

    def fetch_us_stocks(self) -> list[dict]:
        if yf is None:
            logger.warning("yfinance not installed; skipping US stock fetch.")
            return []

        logger.info("Fetching US stock data...")
        results: list[dict] = []
        for symbol in self.us_tickers:
            try:
                ticker = yf.Ticker(symbol)
                quote = self._quote_from_history(symbol, ticker.history(period="1mo"), source="yfinance")
                if quote is not None:
                    results.append(quote)
            except Exception as exc:  # pragma: no cover - live source failures
                logger.error("Error fetching US stock %s: %s", symbol, exc)
        return results

    def fetch_ticker_news(self, *, limit_per_symbol: int = 4) -> list[dict]:
        """Per-symbol headlines from Yahoo Finance; needs no API key, so it
        still yields stock news when NEWS_API_KEY is missing or rate limited."""
        if yf is None or not Config.ENABLE_TICKER_NEWS:
            return []

        logger.info("Fetching per-ticker news...")
        results: list[dict] = []
        for symbol in self.us_tickers:
            try:
                raw_items = yf.Ticker(symbol).news or []
            except Exception as exc:  # pragma: no cover - live source failures
                logger.warning("Ticker news fetch failed for %s: %s", symbol, exc)
                continue
            picked = 0
            for raw in raw_items:
                item = self._normalize_ticker_news(symbol, raw)
                if item is None:
                    continue
                results.append(item)
                picked += 1
                if picked >= limit_per_symbol:
                    break
        return results

    def _normalize_ticker_news(self, symbol: str, raw: dict) -> dict | None:
        # yfinance >= 0.2.50 nests fields under "content"; older releases are flat.
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        title = (content.get("title") or "").strip()
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or content.get("link")
            or ""
        )
        if not title or not url:
            return None

        published_at = content.get("pubDate") or content.get("displayTime")
        if not published_at and content.get("providerPublishTime"):
            published_at = datetime.fromtimestamp(int(content["providerPublishTime"]), tz=timezone.utc).isoformat()
        provider = content.get("provider")
        source_name = provider.get("displayName") if isinstance(provider, dict) else content.get("publisher")
        return {
            "title": title,
            "url": url,
            "desc": (content.get("summary") or content.get("description") or "").strip(),
            "source_name": source_name or "Yahoo Finance",
            "source_type": "news",
            "published_at": published_at,
            "tags": [symbol],
        }

    def _quote_from_history(self, symbol: str, hist, *, source: str) -> dict | None:
        if hist is None or hist.empty:
            return None

        closes = [self._safe_float(value) for value in hist["Close"].tolist()]
        current = closes[-1]
        if current <= 0:
            return None
        prev_close = closes[-2] if len(closes) >= 2 else current
        open_price = self._safe_float(hist["Open"].iloc[-1], default=current)
        low = self._safe_float(hist["Low"].iloc[-1], default=current)
        high = self._safe_float(hist["High"].iloc[-1], default=current)
        change = current - prev_close

        quote = {
            "symbol": symbol,
            "price": round(current, 2),
            "change": f"{change:+.2f}",
            "change_pct": self._pct(current, prev_close),
            "range": f"{low:.2f}-{high:.2f}",
            "open": round(open_price, 2),
            "close": round(current, 2),
            "low": round(low, 2),
            "high": round(high, 2),
            "source": source,
        }
        if len(closes) >= 6:
            quote["change_5d_pct"] = self._pct(current, closes[-6])
        if len(closes) >= 2:
            quote["change_1m_pct"] = self._pct(current, closes[0])
            quote["range_1m"] = f"{min(self._safe_float(v) for v in hist['Low'].tolist()):.2f}-{max(self._safe_float(v) for v in hist['High'].tolist()):.2f}"
        if "Volume" in hist:
            volumes = [self._safe_float(value) for value in hist["Volume"].tolist()]
            quote["volume"] = int(volumes[-1])
            prior = [value for value in volumes[:-1] if value > 0]
            if prior and volumes[-1] > 0:
                # >1 means today's volume runs above the 1-month average: a
                # quick "is something happening" signal for the analyzer.
                quote["volume_ratio"] = round(volumes[-1] / (sum(prior) / len(prior)), 2)
        return quote

    def _pct(self, current: float, base: float) -> float | None:
        if not base:
            return None
        return round((current - base) / base * 100, 2)

    def fetch_tw_stocks(self) -> list[dict]:
        logger.info("Fetching TW stock data...")
        results: list[dict] = []
        for symbol in self.tw_tickers:
            quote = self._fetch_tw_quote(symbol)
            if quote is not None:
                results.append(quote)
        return results

    def _fetch_tw_quote(self, symbol: str) -> dict | None:
        for source in self.tw_source_order:
            if source == "mis":
                quote = self._fetch_tw_from_mis(symbol)
            elif source == "yfinance":
                quote = self._fetch_tw_from_yfinance(symbol)
            else:
                logger.warning("Unknown TW stock source %s for %s.", source, symbol)
                continue

            if quote is not None:
                return quote
        logger.error("All TW stock sources failed for %s.", symbol)
        return None

    def _fetch_tw_from_mis(self, symbol: str) -> dict | None:
        url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_{symbol}.tw"
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Intel-Flow-Bot"},
            )
            response.raise_for_status()
            payload = response.json()
            if "msgArray" not in payload or not payload["msgArray"]:
                return None

            data = payload["msgArray"][0]
            current = self._safe_float(data.get("z") or data.get("y"), default=0.0)
            prev = self._safe_float(data.get("y"), default=current)
            low = self._safe_float(data.get("l"), default=current)
            high = self._safe_float(data.get("h"), default=current)
            open_price = self._safe_float(data.get("o"), default=current)
            if current <= 0:
                return None

            change = current - prev
            return {
                "symbol": symbol,
                "price": round(current, 2),
                "change": f"{change:+.2f}",
                "change_pct": self._pct(current, prev),
                "range": f"{low:.2f}-{high:.2f}",
                "open": round(open_price, 2),
                "close": round(current, 2),
                "low": round(low, 2),
                "high": round(high, 2),
                "source": "mis.twse",
            }
        except Exception as exc:  # pragma: no cover - live source failures
            logger.warning("MIS fetch failed for %s: %s", symbol, exc)
            return None

    def _fetch_tw_from_yfinance(self, symbol: str) -> dict | None:
        if yf is None:
            logger.warning("yfinance not installed; skipping fallback for %s.", symbol)
            return None

        try:
            ticker = yf.Ticker(f"{symbol}.TW")
            return self._quote_from_history(symbol, ticker.history(period="1mo"), source="yfinance")
        except Exception as exc:  # pragma: no cover - live source failures
            logger.warning("yfinance fallback failed for %s: %s", symbol, exc)
            return None

    def _safe_float(self, value, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
