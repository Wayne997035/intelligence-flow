from notion_client import Client
from src.config import Config
from datetime import datetime
from src.utils.logger import logger
import re

class NotionSender:
    def __init__(self):
        self.notion = Client(auth=Config.NOTION_TOKEN) if Config.NOTION_TOKEN else None

    def create_stock_insight_report(self, analysis_text):
        """建立 [投資情報] 完整報告"""
        if not self.notion or analysis_text.startswith("ERROR"): return None
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = f"[投資情報] {now_str}"
        
        try:
            blocks = self._parse_and_build_blocks(analysis_text, "投資與產業分析報告", "blue_background")
            new_page = self.notion.pages.create(
                parent={"database_id": Config.NOTION_PAGE_ID},
                properties={"Name": {"title": [{"text": {"content": title}}]}},
                children=blocks
            )
            logger.info("Detailed Stock Insight report created in Notion.")
            return new_page['url']
        except Exception as e:
            logger.error(f"Notion Stock report failed: {e}")
            return None

    def create_ai_tech_report(self, analysis_text):
        """建立 [AI 技術] 完整報告"""
        if not self.notion or analysis_text.startswith("ERROR"): return None
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = f"[AI 技術] {now_str}"
        
        try:
            blocks = self._parse_and_build_blocks(analysis_text, "AI 前沿技術觀察報告", "gray_background")
            new_page = self.notion.pages.create(
                parent={"database_id": Config.NOTION_PAGE_ID},
                properties={"Name": {"title": [{"text": {"content": title}}]}},
                children=blocks
            )
            logger.info("Detailed AI Tech report created in Notion.")
            return new_page['url']
        except Exception as e:
            logger.error(f"Notion AI Tech report failed: {e}")
            return None

    def _parse_and_build_blocks(self, text, main_heading, bg_color):
        """解析 AI 的結構化輸出並轉化為豐富的 Notion Blocks，隱藏 URL"""
        blocks = [
            {"object": "block", "type": "heading_2", "heading_2": {
                "rich_text": [{"text": {"content": main_heading}}],
                "color": bg_color
            }},
            {"object": "block", "type": "divider", "divider": {}}
        ]

        lines = text.split('\n')
        current_item = {}
        section_summary = ""
        footer_view = ""
        
        # 逐行解析狀態機
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 結算 item
            if line.startswith('[') and current_item.get('title') and current_item.get('url'):
                self._append_item_blocks(blocks, current_item, bg_color)
                current_item = {}

            if line.startswith('[SECTION_SUMMARY]'):
                pass # 標記開始抓取摘要
            elif line.startswith('[EXPERT_VIEW]') or line.startswith('[FUTURE_OUTLOOK]'):
                pass # 標記開始抓取總結
            elif line.startswith('[NEWS_ITEM]') or line.startswith('[TECH_ITEM]'):
                current_item = {}
            elif line.startswith('TITLE:'):
                current_item['title'] = line.replace('TITLE:', '').strip()
            elif line.startswith('URL:'):
                current_item['url'] = line.replace('URL:', '').strip()
            elif line.startswith('SUMMARY:'):
                current_item['summary'] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('INSIGHT:'):
                current_item['insight'] = line.replace('INSIGHT:', '').strip()
            elif not line.startswith('['):
                # 處理非標籤行的文字歸屬
                if "📌" not in text.split(line)[0] and not section_summary:
                    section_summary = line
                elif "[EXPERT_VIEW]" in text.split(line)[0] or "[FUTURE_OUTLOOK]" in text.split(line)[0]:
                    footer_view += line + " "

        # 處理最後一個 item
        if current_item.get('title') and current_item.get('url'):
            self._append_item_blocks(blocks, current_item, bg_color)

        # 插入摘要（放在標題後）
        if section_summary:
            blocks.insert(2, {"object": "block", "type": "paragraph", "paragraph": {
                "rich_text": [{"text": {"content": section_summary}}],
                "color": "gray"
            }})

        # 插入頁尾觀點
        if footer_view:
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            blocks.append({"object": "block", "type": "quote", "quote": {
                "rich_text": [{"text": {"content": f"🎯 總結展望: {footer_view.strip()}"}}]
            }})
            
        return blocks

    def _append_item_blocks(self, blocks, item, bg_color):
        """輔助方法：將單個 Item 加入 Notion Blocks"""
        title = item['title']
        url = item['url']
        summary = item.get('summary', '')
        insight = item.get('insight', '')

        # 1. 標題 (改為純文字，移除連結，視覺更乾淨)
        blocks.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": [
            {"text": {"content": title}}
        ]}})
        
        # 2. 摘要
        if summary:
            blocks.append({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [
                {"text": {"content": summary}}
            ]}})
            
        # 3. 洞察 (Callout + 查看原文)
        if insight:
            blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [
                        {"text": {"content": f"💡 深度洞察: {insight}\n\n"}},
                        {"text": {"content": "🔗 查看原文", "link": {"url": url}}, "annotations": {"italic": True, "bold": True, "color": "blue"}}
                    ],
                    "icon": {"emoji": "💡"},
                    "color": "blue_background" if "blue" in bg_color else "gray_background"
                }
            })
        blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": []}}) # 空行
