import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # AI Analysis (Supports Gemini & Groq)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")
    
    # Social & News
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Deliverers
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    NOTION_TOKEN = os.getenv("NOTION_TOKEN") or os.getenv("NOTION_INTEGRATION_SECRET")
    NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID") or os.getenv("NOTION_DATABASE_ID")
    
    # Stock Lists
    US_STOCKS = ["NVDA", "TSLA", "AMD", "GOOG", "AAPL"]
    TW_STOCKS = ["0050", "2330", "00692"] # 依照用戶要求更新
    
    # Scheduler Config
    INTERVAL_MINUTES = 15
