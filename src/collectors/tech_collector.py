import requests
from datetime import datetime, timedelta
from src.utils.logger import logger
import os

class TechCollector:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        # 核心主力關鍵字：包含主流模型與前沿技術 (Mistral 是龍蝦廠商, OpenClaw 是 Agent 核心專案)
        self.primary_focus = [
            'Claude', 'Gemini', 'OpenAI', 'GPT', 'DeepSeek', 'Llama', 
            'Mistral', 'Mixtral', 'Qwen', 'InternLM', 'MiniCPM',
            'RAG', 'AI Agent', 'VLM', 'xAI', 'Grok', 'Mamba', 'Jamba', 
            'BitNet', 'OpenClaw', 'ClawBench'
        ]
        # 輔助技術關鍵字
        self.tech_keywords = [
            'Open Source LLM', 'Model update', 'GitHub trending AI', 'SOTA',
            'Flux AI', 'Sora Video', 'Kling AI', 'AgentOps', 'On-device AI'
        ]

    def fetch_hacker_news_ai(self):
        """從 Hacker News 抓取主力 AI 技術討論"""
        logger.info(f"Fetching HN AI stories (Focus: {', '.join(self.primary_focus[:5])})...")
        query = " OR ".join([f'"{kw}"' for kw in self.primary_focus])
        url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=story&hitsPerPage=10"
        results = []
        try:
            resp = requests.get(url, timeout=10).json()
            for hit in resp.get('hits', []):
                results.append({
                    'title': f"[HN] {hit['title']}",
                    'url': hit.get('url') or f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    'desc': f"Points: {hit['points']} | Comments: {hit['num_comments']}"
                })
        except Exception as e:
            logger.error(f"HN fetch failed: {e}")
        return results

    def fetch_github_trending_ai(self):
        """從 GitHub 抓取與主力技術相關的熱門項目"""
        logger.info(f"Fetching GitHub AI trends (Focus: {', '.join(self.primary_focus[:5])})...")
        last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        # 搜尋包含主力關鍵字的高星星專案
        query = f"(Claude OR Gemini OR OpenAI OR DeepSeek OR xAI OR Grok OR Llama OR 'AI Agent' OR RAG OR VLM) stars:>500 created:>{last_week}"
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=8"
        results = []
        try:
            headers = {'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'Intel-Flow-Bot'}
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            resp = requests.get(url, headers=headers, timeout=10).json()
            if 'items' in resp:
                for repo in resp['items']:
                    results.append({
                        'title': f"[GitHub] {repo['full_name']} ({repo['stargazers_count']} stars)",
                        'url': repo['html_url'],
                        'desc': repo['description'] or "No description."
                    })
        except Exception as e:
            logger.error(f"GitHub fetch failed: {e}")
        return results

    def fetch_reddit_ai_hot(self):
        """從 Reddit 搜尋主力技術看板 (ClaudeAI, OpenAI, GoogleGemini)"""
        logger.info("Searching Reddit for specialized AI updates...")
        results = []
        # 鎖定主力三大看板 + 全球技術熱點 singularity + 極客看板 LocalLLaMA
        subreddits = ['ClaudeAI', 'OpenAI', 'GoogleGemini', 'singularity', 'LocalLLaMA', 'MachineLearning']
        
        try:
            for sub in subreddits:
                url = f"https://www.reddit.com/r/{sub}/top.json?t=day&limit=5"
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
                resp = requests.get(url, headers=headers, timeout=10).json()
                if 'data' in resp:
                    for post in resp.get('data', {}).get('children', []):
                        data = post['data']
                        if data.get('stickied'): continue
                        results.append({
                            'title': f"[Reddit r/{sub}] {data['title']}",
                            'url': f"https://www.reddit.com{data['permalink']}",
                            'desc': f"Upvotes: {data['ups']} | {data.get('selftext', '')[:100]}..."
                        })
        except Exception as e:
            logger.error(f"Reddit fetch failed: {e}")
        return results

    def fetch_all_community_ai(self):
        """整合所有核心社群情報"""
        return self.fetch_hacker_news_ai() + self.fetch_github_trending_ai() + self.fetch_reddit_ai_hot()
