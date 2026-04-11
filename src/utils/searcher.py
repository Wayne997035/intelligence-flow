import requests
from src.utils.logger import logger

def search_web(query):
    # 使用簡易搜尋模擬器
    logger.info(f'Searching web for: {query}')
    return f'搜尋結果: 關於 {query} 的最新資訊已自動整理。'