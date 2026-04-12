import requests
import xml.etree.ElementTree as ET
from src.utils.logger import logger

class ArxivCollector:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"

    def fetch_latest_ai_papers(self, limit=20):
        """抓取 arXiv 最新 AI 相關論文 (類別 + 關鍵字)"""
        logger.info(f"Fetching top {limit} targeted AI papers from arXiv...")
        # 結合類別篩選與核心關鍵字 (Agent, Optimization, Edge AI)
        query = '(cat:cs.AI OR cat:cs.LG OR cat:cs.CL) AND (all:"AI Agent" OR all:"Foundation Model" OR all:"Edge AI")'
        url = f"{self.base_url}?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={limit}"
        
        results = []
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            
            root = ET.fromstring(resp.text)
            # arXiv API uses Atom format
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                link = entry.find('atom:id', ns).text
                published = entry.find('atom:published', ns).text
                
                results.append({
                    'title': f"[arXiv] {title}",
                    'url': link,
                    'desc': f"Authors: {', '.join([a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)][:3])} | Published: {published[:10]} | Abstract: {summary[:250]}..."
                })
        except Exception as e:
            logger.error(f"arXiv fetch failed: {e}")
            
        return results

    def fetch_all_arxiv(self):
        """整合所有 arXiv 情報"""
        return self.fetch_latest_ai_papers()
