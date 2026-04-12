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

def deduplicate_and_sort(data_list, priority_kws, limit=15):
    """進階去重：比對 URL 與標題前綴，並嚴格限制數量以節省 Token"""
    unique_data = {}
    for item in data_list:
        url = item.get('url')
        title_prefix = item.get('title', '')[:12].strip().upper()
        if not url: continue
        
        # 鍵值結合網址與標題前綴，強力去重
        key = f"{url}_{title_prefix}"
        if key not in unique_data or len(item.get('desc', '')) > len(unique_data[key].get('desc', '')):
            unique_data[key] = item
    
    final_list = list(unique_data.values())
    
    # 關鍵字排序：命中優先級越高的排越前面
    def get_priority(item):
        t_upper = item.get('title', '').upper()
        for i, kw in enumerate(priority_kws):
            if kw.upper() in t_upper:
                return i
        return 999
    
    final_list.sort(key=get_priority)
    
    # 嚴格只取前 limit 則，確保不爆 Token，Notion 也剛好放得下
    return final_list[:limit]

def run_job():
    logger.info("🚀 Intel-Flow 執行循環開始 (精簡高質模式)...")
    
    stock_fetcher = StockCollector()
    news_fetcher = NewsCollector()
    tech_fetcher = TechCollector()
    hf_fetcher = HFCollector()
    arxiv_fetcher = ArxivCollector()
    analyzer = AIAnalyzer()
    discord = DiscordSender()
    notion = NotionSender()
    
    try:
        # 1. 抓取資料
        us_stocks = stock_fetcher.fetch_us_stocks()
        tw_stocks = stock_fetcher.fetch_tw_stocks()
        stock_news_raw = news_fetcher.fetch_stock_news()
        
        # 2. 抓取 AI 技術資料
        ai_official_news = news_fetcher.fetch_ai_tech_news()
        ai_community_data = tech_fetcher.fetch_all_community_ai()
        hf_data = hf_fetcher.fetch_all_hf()
        arxiv_data = arxiv_fetcher.fetch_all_arxiv()
        
        # 處理 AI 技術資料：去重、排序、截斷描述
        raw_ai_data = ai_official_news + ai_community_data + hf_data + arxiv_data
        for item in raw_ai_data:
            if len(item.get('desc', '')) > 150:
                item['desc'] = item['desc'][:150] + "..."
        
        priority_kws = ['Gemma-4', 'GLM-5', 'OpenClaw', 'ClawBench', 'DeepSeek', 'Mistral', 'MiniCPM', 'Mamba', 'BitNet', 'NVDA', 'TSMC']
        all_ai_tech_data = deduplicate_and_sort(raw_ai_data, priority_kws, limit=15)
        
        # 處理股票新聞
        for item in stock_news_raw:
            if len(item.get('desc', '')) > 150:
                item['desc'] = item['desc'][:150] + "..."
        stock_news = deduplicate_and_sort(stock_news_raw, Config.US_STOCKS + Config.TW_STOCKS, limit=12)
        
        # 3. 處理股票報告 (一魚兩吃)
        stocks_info = "\n".join([f"{s['symbol']}: {s['price']} ({s['change']})" for s in us_stocks + tw_stocks])
        stock_report = analyzer.analyze_stock_market(stocks_info, stock_news, is_full_report=True)
        
        if not stock_report.startswith("ERROR"):
            notion_url = notion.create_stock_insight_report(stock_report)
            discord.send_stock_and_analysis(us_stocks, tw_stocks, stock_report, notion_url)
            logger.info("✅ 第一篇：股票與投資分析發送完成。")
        
        logger.info("等待 20 秒緩衝，預防 Token 頻率限制...")
        time.sleep(20)
        
        # 4. 處理 AI 技術報告 (一魚兩吃)
        ai_report = analyzer.analyze_ai_tech(all_ai_tech_data, is_full_report=True)
        
        if not ai_report.startswith("ERROR"):
            notion_url = notion.create_ai_tech_report(ai_report)
            discord.send_ai_tech_report(ai_report, notion_url)
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
        scheduler = BlockingScheduler()
        # 設定每天早上 8 點與晚上 8 點執行
        scheduler.add_job(run_job, 'cron', hour='8,20', minute=0, id='intel_flow_job')
        
        logger.info("⏰ 排程模式已啟動：每天 08:00 與 20:00 自動執行。")
        logger.info("💡 請保持此視窗開啟以持續運行...")
        
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("👋 排程已手動停止。")
    else:
        run_job()
