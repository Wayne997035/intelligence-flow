import requests
from datetime import datetime, timedelta
from src.config import Config
from src.utils.logger import logger

class NewsCollector:
    def __init__(self):
        self.api_key = Config.NEWS_API_KEY
        
    def fetch_stock_news(self):
        """根據關注的股票清單抓取相關的高質量新聞"""
        domains = "reuters.com,bloomberg.com,wsj.com,cnbc.com,techcrunch.com,finance.yahoo.com"
        # 動態生成關鍵字：股票代碼 + 半導體/AI 供應鏈關鍵字
        stock_kws = Config.US_STOCKS + Config.TW_STOCKS + [
            'Nvidia Blackwell', 'TSMC 2nm', 'AI chip demand', 
            'Semiconductor supply chain', 'Earnings report', 'Price target'
        ]
        return self._fetch_by_keywords(stock_kws, domains=domains, pageSize=10, days_back=7)

    def fetch_ai_tech_news(self):
        """抓取全球最新發布的模型、SOTA 論文與 AI 突破"""
        domains = "openai.com,anthropic.com,github.blog,googleblog.com,techcrunch.com,venturebeat.com,theverge.com,arstechnica.com"
        keywords = [
            'New Model Release', 'SOTA', 'Foundation Model', 
            'Claude 3.5', 'GPT-5', 'Gemini 2.0', 'Llama 4',
            'Mistral AI', 'Mixtral', 'Qwen 2.5', 'InternLM',
            'MiniCPM', 'AI Agent Architecture', 'OpenClaw', 
            'Edge AI', 'BitNet', 'Mamba architecture', 'Video generation AI'
        ]
        return self._fetch_by_keywords(keywords, domains=domains, pageSize=15, days_back=3)
            
    def _fetch_by_keywords(self, keywords, domains=None, pageSize=10, days_back=5):
        if not self.api_key:
            logger.warning("News API Key missing.")
            return []
            
        all_news = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # 使用更嚴格的搜尋語法
        query = "(" + " OR ".join([f'"{kw}"' for kw in keywords]) + ")"
        
        try:
            params = {
                'apiKey': self.api_key,
                'q': query,
                'from': from_date,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 20  # 設定為 20 則
            }
            if domains:
                params['domains'] = domains
                
            url = f"https://newsapi.org/v2/everything"
            res = requests.get(url, params=params, timeout=10).json()
            
            if 'articles' in res:
                for a in res['articles']:
                    # 放寬過濾條件：只要標題或內容有相關關鍵字就保留，不再使用嚴格白名單
                    all_news.append({
                        'title': a['title'], 
                        'url': a['url'], 
                        'desc': a.get('description', '') or ''
                    })
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
        return all_news
