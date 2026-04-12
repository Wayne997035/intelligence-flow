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

    def analyze_stock_market(self, stocks_info, news, is_full_report=True):
        news_str = "\n".join([f"新聞 {i+1}:\n標題: {n['title']}\n來源網址: {n['url']}\n摘要: {n['desc']}" for i, n in enumerate(news)])
        
        prompt = f"""
        【重要指令：必須使用「繁體中文(台灣)」回答，嚴禁使用簡體字與大陸術語(如：不得使用芯片、服務器、軟件、性能等)】
        
        你是一位專門追蹤 {Config.US_STOCKS} 與 {Config.TW_STOCKS} 的資深投資分析師。
        請根據以下報價與新聞撰寫詳盡報告。
        
        當前報價：{stocks_info}
        待分析新聞：
        {news_str}
        
        分析要求：
        1. 針對每家公司結合新聞進行深度點評，分析對股價的具體利多或利空。
        2. 洞察必須包含專業的基本面支撐邏輯。
        
        格式：
        [SECTION_SUMMARY] (整體市場摘要)
        [NEWS_ITEM] TITLE: (標題) URL: (來源網址) SUMMARY: (摘要) INSIGHT: (專業深度洞察)
        [EXPERT_VIEW] (總結與建議)
        """
        return self._get_ai_response(prompt)

    def analyze_ai_tech(self, news, is_full_report=True):
        news_str = "\n".join([f"技術情報 {i+1}:\n原始標題: {n['title']}\n來源網址: {n['url']}\n細節: {n['desc']}" for i, n in enumerate(news)])
        
        prompt = f"""
        【重要指令：必須使用「繁體中文(台灣)」回答，絕對嚴禁使用簡體字。技術術語必須使用台灣習慣用語。】
        
        你是一位極客風格的 AI 技術架構師。你的任務是從以下情報中分析最新出的、最具影響力的技術。
        情報清單：
        {news_str}
        
        分析要求：
        1. TITLE 必須保留完整版本號(如 Gemma-4-31B-it, GLM-5.1)。
        2. 必須包含每一項新出的模型、硬核研究(arXiv)或 Agent 運維工具。
        3. SUMMARY 請寫出核心規格；INSIGHT 請寫出架構分析與開發者價值。
        
        格式：
        [SECTION_SUMMARY] (技術演進摘要)
        [TECH_ITEM] TITLE: (名稱/型號) URL: (來源網址) SUMMARY: (技術細節) INSIGHT: (深度評析)
        [FUTURE_OUTLOOK] (未來預測)
        """
        return self._get_ai_response(prompt)

    def _get_ai_response(self, prompt):
        # 1. 優先嘗試 Gemini
        if self.gemini_client:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
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
