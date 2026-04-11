import requests
from bs4 import BeautifulSoup
from src.utils.logger import logger

def fetch_web_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return "\n".join(paragraphs)[:2000]
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return ""
