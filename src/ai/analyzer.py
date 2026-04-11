from google import genai
from openai import OpenAI
from src.config import Config
from src.utils.logger import logger

class AIAnalyzer:
    def __init__(self):
        self.openai_client = None
        self.gemini_client = None
        
        if Config.OPENAI_API_KEY and not Config.OPENAI_API_KEY.startswith("AIzaSy"):
            logger.info("Using OpenAI (GPT-4o) for analysis.")
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
        elif Config.OPENAI_API_KEY and Config.OPENAI_API_KEY.startswith("AIzaSy"):
            logger.info("Using Google Gemini (New SDK) for analysis.")
            self.gemini_client = genai.Client(api_key=Config.OPENAI_API_KEY)
        else:
            logger.warning("No valid AI API Key found.")

    def analyze_stock_market(self, stocks_info, news):
        """分析股票及相關產業新聞，要求 AI 附上 URL"""
        if not self.openai_client and not self.gemini_client:
            return "ERROR: 無法調用 AI API"
        
        # 建立新聞對照表，讓 AI 可以附上正確的 URL
        news_str = "\n".join([f"新聞 {i+1}:\n標題: {n['title']}\n來源連結: {n['url']}\n摘要: {n['desc']}" for i, n in enumerate(news)])
        
        prompt = f"""
        你是一位資深科技投資分析師。請分析以下股票數據與相關產業新聞。
        
        股票數據概況：
        {stocks_info}
        
        待分析新聞 (請確實引用新聞中的來源連結)：
        {news_str}
        
        請嚴格按照以下格式輸出：
        
        [SECTION_SUMMARY]
        (整體市場摘要)
        
        [NEWS_ITEM]
        TITLE: (標題)
        URL: (來源連結)
        SUMMARY: (1-2 行摘要)
        INSIGHT: (深度洞察)
        
        [EXPERT_VIEW]
        (最終總結)
        """
        return self._get_ai_response(prompt)

    def analyze_ai_tech(self, news):
        """分析最新的 AI 技術知識，要求 AI 附上 URL"""
        if not self.openai_client and not self.gemini_client:
            return "ERROR: 無法調用 AI API"
        
        news_str = "\n".join([f"技術情報 {i+1}:\n標題: {n['title']}\n來源連結: {n['url']}\n細節: {n['desc']}" for i, n in enumerate(news)])
        
        prompt = f"""
        你是一位專注於前沿技術的 AI 觀察家。請分析以下最新的模型動態、爆紅開源項目與學術論文。
        特別注意 Hugging Face 上的趨勢模型與 Daily Papers。
        
        技術情報清單 (請確實引用對應的來源連結)：
        {news_str}
        
        請嚴格按照以下格式輸出：
        
        [SECTION_SUMMARY]
        (AI 技術發展概況摘要)
        
        [TECH_ITEM]
        TITLE: (技術或模型名稱)
        URL: (對應的來源連結)
        SUMMARY: (更新重點)
        INSIGHT: (影響分析)
        
        [FUTURE_OUTLOOK]
        (未來趨勢總結)
        """
        return self._get_ai_response(prompt)

    def _get_ai_response(self, prompt):
        try:
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=Config.AI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content
            elif self.gemini_client:
                # 使用新版 SDK 的調用方式，並設定為 gemini-flash-latest 以自動對應最新穩定版本
                model_name = Config.AI_MODEL if not Config.AI_MODEL.startswith("gpt") else "gemini-flash-latest"
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text if hasattr(response, 'text') else "ERROR: AI 回傳內容為空"
        except Exception as e:
            logger.error(f"AI Analysis failed: {e}")
            return f"ERROR: {str(e)}"
