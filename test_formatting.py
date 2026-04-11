import os
import sys
import re
from src.deliverers.discord_sender import DiscordSender

def test_formatting():
    sender = DiscordSender()
    
    # 模擬 AI 的輸出格式 (採用新的 TITLE-URL-SUMMARY-INSIGHT 順序)
    mock_ai_output = """
[SECTION_SUMMARY]
當前市場由 AI 需求驅動，晶片相關個股表現優異。

[NEWS_ITEM]
TITLE: 台積電 Q1 營收創歷史新高
URL: https://www.cnbc.com/2026/04/10/tsmc-q1-record-revenue-ai-chip-demand-strong.html
SUMMARY: 受惠於 Apple 和 Nvidia 的強勁需求。
INSIGHT: 確信台積電在 AI 供應鏈的核心地位。

[EXPERT_VIEW]
科技巨頭仍保持上漲態勢，信心穩固。
"""
    
    formatted = sender._format_text_with_links(mock_ai_output)
    print("--- 格式化後的結果 ---")
    print(formatted)
    
    # 驗證是否包含文字超連結且隱藏了原始 URL
    assert "[台積電 Q1 營收創歷史新高](https://www.cnbc.com/2026/04/10/tsmc-q1-record-revenue-ai-chip-demand-strong.html)" in formatted
    
    # 確保沒有單獨的 URL 標籤出現在最終結果中
    assert "URL: https://" not in formatted
    print("\n✅ 格式化測試通過！")

if __name__ == "__main__":
    test_formatting()
