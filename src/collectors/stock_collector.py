from __future__ import annotations

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
                info = ticker.fast_info
                hist = ticker.history(period="1d")
                if hist.empty:
                    continue
                open_price = self._safe_float(hist["Open"].iloc[0])
                low = self._safe_float(hist["Low"].iloc[0])
                high = self._safe_float(hist["High"].iloc[0])
                close = self._safe_float(hist["Close"].iloc[-1])
                prev_close = self._safe_float(getattr(info, "previous_close", None), default=close)
                current = close
                change = current - prev_close
                results.append(
                    {
                        "symbol": symbol,
                        "price": round(current, 2),
                        "change": f"{change:+.2f}",
                        "range": f"{low:.2f}-{high:.2f}",
                        "open": round(open_price, 2),
                        "close": round(close, 2),
                        "low": round(low, 2),
                        "high": round(high, 2),
                        "source": "yfinance",
                    }
                )
            except Exception as exc:  # pragma: no cover - live source failures
                logger.error("Error fetching US stock %s: %s", symbol, exc)
        return results

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
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            open_price = self._safe_float(hist["Open"].iloc[-1])
            low = self._safe_float(hist["Low"].iloc[-1])
            high = self._safe_float(hist["High"].iloc[-1])
            current = self._safe_float(hist["Close"].iloc[-1])
            prev = self._safe_float(getattr(ticker.fast_info, "previous_close", None), default=current)
            change = current - prev
            return {
                "symbol": symbol,
                "price": round(current, 2),
                "change": f"{change:+.2f}",
                "range": f"{low:.2f}-{high:.2f}",
                "open": round(open_price, 2),
                "close": round(current, 2),
                "low": round(low, 2),
                "high": round(high, 2),
                "source": "yfinance",
            }
        except Exception as exc:  # pragma: no cover - live source failures
            logger.warning("yfinance fallback failed for %s: %s", symbol, exc)
            return None

    def _safe_float(self, value, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
