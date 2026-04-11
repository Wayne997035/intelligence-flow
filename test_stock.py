from src.collectors.stock_collector import StockCollector
stocks = ['NVDA', 'TSLA', 'AMD', 'GOOG']
print(StockCollector(stocks).fetch())