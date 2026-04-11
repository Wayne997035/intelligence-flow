import requests
from bs4 import BeautifulSoup

class WebCollector:
    def __init__(self):
        # 這些是穩定更新的新聞頁面
        self.sources = {
            'SpaceX': 'https://www.spacex.com/updates/',
            'Anthropic': 'https://www.anthropic.com/news'
        }
    
    def fetch_all(self):
        results = {}
        for name, url in self.sources.items():
            try:
                res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                soup = BeautifulSoup(res.text, 'html.parser')
                # 簡單提取標題 (後續若結構複雜，我們再細調選擇器)
                items = [a.get_text().strip() for a in soup.find_all('h3', limit=3)]
                results[name] = items if items else ["暫無更新"]
            except Exception:
                results[name] = ["無法獲取內容"]
        return results
