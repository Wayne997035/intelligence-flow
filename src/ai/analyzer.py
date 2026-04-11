from google import genai
from groq import Groq
from src.config import Config
from src.utils.logger import logger
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

class AIAnalyzer:
    def __init__(self):
        self.gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY) if Config.GEMINI_API_KEY else None
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
        logger.info("Analyzer initialized with Gemini (Primary) and Groq (Fallback).")

    def analyze_stock_market(self, stocks_info, news):
        news_str = "\n".join([f"新聞 {i+1}:\n標題: {n['title']}\n來源網址: {n['url']}\n摘要: {n['desc']}" for i, n in enumerate(news)])
        prompt = f"""
        你是一位資深科技投資分析師。請分析以下股票數據與相關產業新聞。
        股票數據概況：{stocks_info}
        待分析新聞清單：
        {news_str}
        
        請嚴格按照以下格式輸出，每個欄位請獨立一行，URL 必須完全複製我提供的「來源網址」：
        [SECTION_SUMMARY]
        (整體市場摘要)

        [NEWS_ITEM]
        TITLE: (標題)
        URL: (請務必精確複製對應新聞的「來源網址」)
        SUMMARY: (摘要)
        INSIGHT: (深度洞察)

        [EXPERT_VIEW]
        (最終總結)
        """
        return self._get_ai_response(prompt)

    def analyze_ai_tech(self, news):
        news_str = "\n".join([f"技術情報 {i+1}:\n標題: {n['title']}\n來源網址: {n['url']}\n細節: {n['desc']}" for i, n in enumerate(news)])
        prompt = f"""
        你是一位極客 (Geek) 風格的 AI 技術架構師與開源觀察家。請分析以下最新的模型動態與技術進展。
        技術情報清單：
        {news_str}
        
        重點觀察方向：
        1. 核心架構突破 (如 Transformer 改進、Mamba 等新架構)。
        2. AI Agent 技能與自主性提升 (Tool Use, Planning, Multi-agent)。
        3. GitHub 熱門項目中的實戰工具、庫與開源模型權重發布。
        4. 學術論文中的前沿趨勢 (SOTA 追蹤)。
        
        請嚴格按照以下格式輸出，每個欄位請獨立一行，URL 必須完全複製我提供的「來源網址」：
        [SECTION_SUMMARY]
        (技術演進與開源社群趨勢概況)

        [TECH_ITEM]
        TITLE: (技術/項目名稱)
        URL: (請務必精確複製對應情報的「來源網址」)
        SUMMARY: (技術亮點/核心邏輯)
        INSIGHT: (架構分析/對開發者的實際影響/Agent 技能點評)

        [FUTURE_OUTLOOK]
        (未來技術趨勢預測與開源競爭格局)
        """
        return self._get_ai_response(prompt)

    def _get_ai_response(self, prompt):
        # 1. 優先嘗試 Gemini
        if self.gemini_client:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                response = self.gemini_client.models.generate_content(
                    model=Config.AI_MODEL,
                    contents=prompt
                )
                signal.alarm(0)
                return response.text
            except Exception as e:
                signal.alarm(0)
                logger.warning(f"Gemini failed (trying Groq): {e}")

        # 2. 備援嘗試 Groq
        if self.groq_client:
            logger.info("Switching to Groq fallback.")
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile"
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Groq fallback failed: {e}")
        
        return "ERROR: 所有 AI 服務皆不可用"
