import yfinance as yf
import requests
import urllib3
from src.config import Config
from src.utils.logger import logger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class StockCollector:
    def __init__(self):
        self.us_tickers = Config.US_STOCKS
        self.tw_tickers = Config.TW_STOCKS

    def fetch_us_stocks(self):
        logger.info("Fetching US stock data...")
        results = []
        for symbol in self.us_tickers:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current = info.last_price
                    prev_close = info.previous_close
                    change = current - prev_close
                    low = hist['Low'].iloc[0]
                    high = hist['High'].iloc[0]
                    results.append({
                        "symbol": symbol,
                        "price": round(current, 2),
                        "change": f"{'+' if change >= 0 else ''}{round(change, 2)}",
                        "range": f"{round(low, 1)}-{round(high, 1)}"
                    })
            except Exception as e:
                logger.error(f"Error fetching US stock {symbol}: {e}")
        return results

    def fetch_tw_stocks(self):
        logger.info("Fetching TW stock data...")
        results = []
        for symbol in self.tw_tickers:
            try:
                # Using TWSE API for real-time TW data
                url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_{symbol}.tw"
                resp = requests.get(url, timeout=10, verify=False).json()
                if "msgArray" in resp and len(resp["msgArray"]) > 0:
                    data = resp["msgArray"][0]
                    current = float(data.get("z", data.get("y", 0)))
                    prev = float(data.get("y", 0))
                    change = current - prev
                    low = float(data.get("l", 0))
                    high = float(data.get("h", 0))
                    results.append({
                        "symbol": symbol,
                        "price": round(current, 2),
                        "change": f"{'+' if change >= 0 else ''}{round(change, 2)}",
                        "range": f"{round(low, 1)}-{round(high, 1)}"
                    })
                else:
                    # Fallback to yfinance if TWSE fails
                    ticker = yf.Ticker(f"{symbol}.TW")
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev = ticker.fast_info.previous_close
                        change = current - prev
                        results.append({
                            "symbol": symbol,
                            "price": round(current, 2),
                            "change": f"{'+' if change >= 0 else ''}{round(change, 2)}",
                            "range": f"{round(hist['Low'].iloc[-1], 1)}-{round(hist['High'].iloc[-1], 1)}"
                        })
            except Exception as e:
                logger.error(f"Error fetching TW stock {symbol}: {e}")
        return results
