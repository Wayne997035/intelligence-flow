import argparse
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from src.collectors.stock_collector import StockCollector
from src.collectors.news_collector import NewsCollector
from src.collectors.tech_collector import TechCollector
from src.collectors.hf_collector import HFCollector
from src.collectors.arxiv_collector import ArxivCollector
from src.ai.analyzer import AIAnalyzer
from src.deliverers.discord_sender import DiscordSender
from src.deliverers.notion_sender import NotionSender
from src.config import Config
from src.utils.logger import logger

def run_job():
    logger.info("🚀 Intel-Flow 執行循環開始 (雙主題 + 多源社群模式)...")
    
    stock_fetcher = StockCollector()
    news_fetcher = NewsCollector()
    tech_fetcher = TechCollector()
    hf_fetcher = HFCollector()
    arxiv_fetcher = ArxivCollector()
    analyzer = AIAnalyzer()
    discord = DiscordSender()
    notion = NotionSender()
    
    try:
        # 1. 抓取股票資料
        us_stocks = stock_fetcher.fetch_us_stocks()
        tw_stocks = stock_fetcher.fetch_tw_stocks()
        stock_news = news_fetcher.fetch_stock_news()
        
        # 2. 抓取 AI 技術資料 (NewsAPI + Hacker News + GitHub + Hugging Face + arXiv)
        ai_official_news = news_fetcher.fetch_ai_tech_news()
        ai_community_data = tech_fetcher.fetch_all_community_ai()
        hf_data = hf_fetcher.fetch_all_hf()
        arxiv_data = arxiv_fetcher.fetch_all_arxiv()
        all_ai_tech_data = ai_official_news + ai_community_data + hf_data + arxiv_data
        
        # 3. 處理第一篇：股票報價 + 產業分析
        stocks_info = "\n".join([f"{s['symbol']}: {s['price']} ({s['change']})" for s in us_stocks + tw_stocks])
        stock_analysis = analyzer.analyze_stock_market(stocks_info, stock_news)
        
        if not stock_analysis.startswith("ERROR"):
            notion_url = notion.create_stock_insight_report(stock_analysis)
            discord.send_stock_and_analysis(us_stocks, tw_stocks, stock_analysis, notion_url)
            logger.info("✅ 第一篇：股票與投資分析發送完成。")
        
        # 4. 處理第二篇：AI 技術前沿 (多源整合)
        ai_tech_analysis = analyzer.analyze_ai_tech(all_ai_tech_data)
        
        if not ai_tech_analysis.startswith("ERROR"):
            notion_url = notion.create_ai_tech_report(ai_tech_analysis)
            discord.send_ai_tech_report(ai_tech_analysis, notion_url)
            logger.info("✅ 第二篇：AI 前沿技術情報發送完成。")
            
    except Exception as e:
        logger.error(f"❌ 執行過程中發生未預期錯誤: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intel-Flow Market Intelligence")
    parser.add_argument("--once", action="store_true", help="執行一次後退出")
    parser.add_argument("--schedule", action="store_true", help=f"每隔 {Config.INTERVAL_MINUTES} 分鐘執行一次")
    
    args = parser.parse_args()
    
    if args.once:
        run_job()
    elif args.schedule:
        logger.info(f"⏰ 排程已啟動：每 {Config.INTERVAL_MINUTES} 分鐘執行一次。")
        scheduler = BlockingScheduler()
        scheduler.add_job(run_job, 'interval', minutes=Config.INTERVAL_MINUTES, next_run_time=time.strftime('%Y-%m-%d %H:%M:%S'))
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("👋 排程已手動停止。")
    else:
        run_job()
