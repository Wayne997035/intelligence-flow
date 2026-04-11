import requests
from datetime import datetime, timedelta
from src.config import Config
from src.utils.logger import logger

class NewsCollector:
    def __init__(self):
        self.api_key = Config.NEWS_API_KEY
        
    def fetch_stock_news(self):
        """抓取與指定股票及半導體產業相關的高質量新聞"""
        # 排除雜訊來源，鎖定權威財經媒體
        domains = "reuters.com,bloomberg.com,wsj.com,cnbc.com,techcrunch.com"
        keywords = ['Nvidia Blackwell', 'TSMC 2nm', 'AI chip demand', 'Semiconductor supply chain']
        return self._fetch_by_keywords(keywords, domains=domains, pageSize=8, days_back=7)

    def fetch_ai_tech_news(self):
        """抓取真正前沿、有討論度的 AI 技術與 GitHub 爆紅項目"""
        # 排除 PyPI, LibHunt 等雜訊，鎖定權威技術媒體與部落格
        domains = "techcrunch.com,venturebeat.com,theverge.com,github.blog,openai.com,anthropic.com,wired.com,arstechnica.com"
        keywords = [
            'Claude AI', 
            'OpenAI o1', 
            'Gemini AI', 
            'Llama AI',
            'DeepSeek AI',
            'AI Agent',
            'RAG technology',
            'VLM',
            'xAI Grok',
            'Codex'
        ]
        return self._fetch_by_keywords(keywords, domains=domains, pageSize=12, days_back=5)
            
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
                'sortBy': 'relevancy', # 優先抓取最相關/熱門的新聞
                'pageSize': pageSize
            }
            if domains:
                params['domains'] = domains
                
            url = f"https://newsapi.org/v2/everything"
            res = requests.get(url, params=params, timeout=10).json()
            
            if 'articles' in res:
                for a in res['articles']:
                    # 再次過濾：確保標題包含我們感興趣的關鍵字
                    title_upper = a['title'].upper()
                    filter_kws = ['CLAUDE', 'OPENAI', 'GEMINI', 'LLAMA', 'DEEPSEEK', 'RAG', 'AGENT', 'AI', 'XAI', 'GROK', 'CODEX', 'NVDA', 'TSMC']
                    if any(kw in title_upper for kw in filter_kws):
                        all_news.append({
                            'title': a['title'], 
                            'url': a['url'], 
                            'desc': a.get('description', '') or ''
                        })
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
        return all_news
