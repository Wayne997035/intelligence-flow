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
        """解析 AI 的結構化輸出並轉化為豐富的 Notion Blocks"""
        import re
        
        blocks = [
            {"object": "block", "type": "heading_2", "heading_2": {
                "rich_text": [{"text": {"content": main_heading}}],
                "color": bg_color
            }},
            {"object": "block", "type": "divider", "divider": {}}
        ]

        # 1. 摘要
        summary_match = re.search(r'\[SECTION_SUMMARY\]\n?(.*?)(?=\n\n?\[|$)', text, re.DOTALL)
        if summary_match:
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {
                "rich_text": [{"text": {"content": summary_match.group(1).strip()}}],
                "color": "gray"
            }})

        # 2. 項目 (NEWS_ITEM 或 TECH_ITEM)
        item_pattern = re.compile(r'\[(?:NEWS_ITEM|TECH_ITEM)\](.*?)(?=\n\n?\[|$)', re.DOTALL)
        items = item_pattern.findall(text)

        def extract_field(pattern, block_text):
            # 尋找該欄位，直到遇到下一個欄位標籤或 block 結束
            match = re.search(pattern + r':\s*(.*?)(?=\s*(?:TITLE|URL|SUMMARY|INSIGHT|\[|$))', block_text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""
        
        for item_content in items:
            t_str = extract_field('TITLE', item_content)
            u_str = extract_field('URL', item_content)
            s_str = extract_field('SUMMARY', item_content)
            i_str = extract_field('INSIGHT', item_content)
            
            if t_str and u_str:
                # 標題
                blocks.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": [
                    {"text": {"content": t_str}}
                ]}})
                
                # 摘要
                if s_str:
                    blocks.append({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [
                        {"text": {"content": s_str}}
                    ]}})
                    
                # 洞察 + 連結
                if i_str:
                    blocks.append({
                        "object": "block",
                        "type": "callout",
                        "callout": {
                            "rich_text": [
                                {"text": {"content": f"深度洞察: {i_str}\n\n"}},
                                {"text": {"content": "🔗 查看原文", "link": {"url": u_str}}, "annotations": {"italic": True, "bold": True, "color": "blue"}}
                            ],
                            "icon": {"emoji": "💡"},
                            "color": "blue_background" if "blue" in bg_color else "gray_background"
                        }
                    })
                blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": []}})

        # 3. 頁尾
        outlook_match = re.search(r'\[(?:FUTURE_OUTLOOK|EXPERT_VIEW)\]\n?(.*?)(?=\n\n?\[|$)', text, re.DOTALL)
        if outlook_match:
            label = "🔮 未來展望" if "[FUTURE_OUTLOOK]" in text else "🕵️ 專家總結"
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            blocks.append({"object": "block", "type": "quote", "quote": {
                "rich_text": [{"text": {"content": f"🎯 {label}: {outlook_match.group(1).strip()}"}}]
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
