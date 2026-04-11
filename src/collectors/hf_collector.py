import requests
from src.utils.logger import logger

class HFCollector:
    def __init__(self):
        self.base_url = "https://huggingface.co/api"

    def fetch_trending_models(self, limit=5):
        """抓取 Hugging Face 當前熱門模型"""
        logger.info(f"Fetching top {limit} trending models from Hugging Face...")
        # 使用新的 trending API
        url = f"https://huggingface.co/api/trending?type=model&limit={limit}"
        results = []
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # trending API 回傳的格式是 {"recentlyTrending": [...]}
            trending_list = data.get('recentlyTrending', [])
            for m in trending_list:
                repo_data = m.get('repoData', {})
                model_id = repo_data.get('id')
                if not model_id: continue
                
                results.append({
                    'title': f"[HF Model] {model_id} ({repo_data.get('likes', 0)} likes)",
                    'url': f"https://huggingface.co/{model_id}",
                    'desc': f"Downloads: {repo_data.get('downloads', 0)} | Task: {repo_data.get('pipeline_tag', 'N/A')}"
                })
        except Exception as e:
            logger.error(f"HF Models fetch failed: {e}")
        return results

    def fetch_daily_papers(self, limit=5):
        """抓取 Hugging Face Daily Papers"""
        logger.info(f"Fetching top {limit} daily papers from Hugging Face...")
        url = f"{self.base_url}/daily_papers?limit={limit}"
        results = []
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            papers = resp.json()
            for p in papers:
                paper = p.get('paper', {})
                title = paper.get('title')
                paper_id = paper.get('id')
                results.append({
                    'title': f"[HF Paper] {title}",
                    'url': f"https://huggingface.co/papers/{paper_id}",
                    'desc': f"Published: {paper.get('publishedAt', 'N/A')} | Summary: {paper.get('summary', '')[:150]}..."
                })
        except Exception as e:
            logger.error(f"HF Papers fetch failed: {e}")
        return results

    def fetch_all_hf(self):
        """整合所有 HF 情報"""
        return self.fetch_trending_models() + self.fetch_daily_papers()
