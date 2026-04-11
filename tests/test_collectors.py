import os
import sys
import unittest
from dotenv import load_dotenv

# 修正路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.collectors.stock_collector import StockCollector
from src.collectors.news_collector import NewsCollector
from src.collectors.tech_collector import TechCollector
from src.collectors.hf_collector import HFCollector

class TestCollectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()

    def test_stock_fetcher(self):
        print("\n--- 測試 StockCollector ---")
        fetcher = StockCollector()
        us = fetcher.fetch_us_stocks()
        tw = fetcher.fetch_tw_stocks()
        print(f"美股抓取數量: {len(us)}")
        print(f"台股抓取數量: {len(tw)}")
        self.assertTrue(len(us) > 0)
        self.assertTrue(len(tw) > 0)

    def test_news_fetcher(self):
        print("\n--- 測試 NewsCollector ---")
        fetcher = NewsCollector()
        stock_news = fetcher.fetch_stock_news()
        ai_news = fetcher.fetch_ai_tech_news()
        print(f"投資新聞數量: {len(stock_news)}")
        print(f"AI 技術新聞數量: {len(ai_news)}")
        for n in ai_news[:2]:
            print(f"範例新聞: {n['title']} (URL: {n['url']})")
        self.assertTrue(len(ai_news) > 0)

    def test_tech_community_fetcher(self):
        print("\n--- 測試 TechCollector (HN & GitHub) ---")
        fetcher = TechCollector()
        hn = fetcher.fetch_hacker_news_ai()
        github = fetcher.fetch_github_trending_ai()
        print(f"HN 熱點數量: {len(hn)}")
        print(f"GitHub 爆紅數量: {len(github)}")
        for g in github[:2]:
            print(f"範例專案: {g['title']} (URL: {g['url']})")
        self.assertTrue(len(github) > 0)

    def test_hf_fetcher(self):
        print("\n--- 測試 HFCollector ---")
        fetcher = HFCollector()
        trending = fetcher.fetch_trending_models()
        papers = fetcher.fetch_daily_papers()
        print(f"HF 熱門模型數量: {len(trending)}")
        print(f"HF 每日論文數量: {len(papers)}")
        for m in trending[:2]:
            print(f"範例模型: {m['title']} (URL: {m['url']})")
        for p in papers[:2]:
            print(f"範例論文: {p['title']} (URL: {p['url']})")
        self.assertTrue(len(trending) > 0)
        self.assertTrue(len(papers) > 0)

if __name__ == '__main__':
    unittest.main()
