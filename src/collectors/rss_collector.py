import feedparser
import requests

class RSSCollector:
    def __init__(self):
        self.sources = {
            'SpaceX': 'https://www.spacex.com/updates/rss.xml',
            'Anthropic': 'https://www.anthropic.com/news/rss'
        }
    
    def fetch_all(self):
        results = {}
        for name, url in self.sources.items():
            try:
                feed = feedparser.parse(url)
                results[name] = [entry.title for entry in feed.entries[:2]]
            except:
                results[name] = ["暫無更新"]
        return results
