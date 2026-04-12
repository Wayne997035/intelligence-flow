import requests
import re
from src.config import Config
from src.utils.logger import logger

class DiscordSender:
    def send_stock_and_analysis(self, us_stocks, tw_stocks, analysis_text, notion_url):
        """發送報價 + 股票分析 (第一篇)"""
        if not Config.DISCORD_WEBHOOK_URL: return
        
        us_lines = "\n\n".join([f"**{s['symbol']}**\n現:{s['price']} | 變:{s['change']}\n區:{s['range']}" for s in us_stocks])
        tw_lines = "\n\n".join([f"**{s['symbol']}**\n現:{s['price']} | 變:{s['change']}\n區:{s['range']}" for s in tw_stocks])

        clean_text_dict = self._format_text_with_links_dict(analysis_text)
        
        description = f"🇺🇸 **美股**\n{us_lines}\n\n----------------\n"
        description += f"🇹🇼 **台股**\n{tw_lines}\n\n"
        description += f"----------------\n"
        
        if clean_text_dict['summary']:
            description += f"📌 **摘要**\n{clean_text_dict['summary']}\n\n"
            
        description += clean_text_dict['items_text']
        
        if clean_text_dict['outlook']:
            description += f"\n{clean_text_dict['outlook_label']}\n{clean_text_dict['outlook']}\n"

        if notion_url:
            description += f"\n📒 [**在 Notion 查看完整深度分析報告**]({notion_url})"

        try:
            payload = {
                "embeds": [{
                    "title": "投資情報報告",
                    "description": description[:4096],
                    "color": 0x2ecc71
                }]
            }
            requests.post(Config.DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            logger.info("Sent Stock Report to Discord.")
        except Exception as e:
            logger.error(f"Stock Report send failed: {e}")

    def send_ai_tech_report(self, ai_text, notion_url):
        """發送 AI 技術前沿情報 (第二篇)"""
        if not Config.DISCORD_WEBHOOK_URL or ai_text.startswith("ERROR"): return
        
        clean_text_dict = self._format_text_with_links_dict(ai_text)
        
        description = ""
        if clean_text_dict['summary']:
            description += f"📌 **摘要**\n{clean_text_dict['summary']}\n\n"
            
        description += clean_text_dict['items_text']
        
        if clean_text_dict['outlook']:
            description += f"\n{clean_text_dict['outlook_label']}\n{clean_text_dict['outlook']}\n"

        if notion_url:
            description += f"\n📒 [**在 Notion 查看完整 AI 技術情報**]({notion_url})"

        try:
            payload = {
                "embeds": [{
                    "title": "AI 技術前沿情報",
                    "description": description[:4096],
                    "color": 0x3498db
                }]
            }
            requests.post(Config.DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            logger.info("Sent AI Tech Report to Discord.")
        except Exception as e:
            logger.error(f"AI Tech Report send failed: {e}")

    def _format_text_with_links_dict(self, text):
        """解析全文並回傳包含各區塊的字典，以便靈活排版"""
        import re
        text = re.sub(r'^-+$', '', text, flags=re.MULTILINE)
        
        summary_match = re.search(r'\[SECTION_SUMMARY\]\s*(.*?)(?=\s*\[|$)', text, re.DOTALL)
        outlook_match = re.search(r'\[(?:FUTURE_OUTLOOK|EXPERT_VIEW)\]\s*(.*?)(?=\s*\[|$)', text, re.DOTALL)
        
        summary = ""
        if summary_match:
            summary = re.sub(r'TITLE:|URL:|SUMMARY:|INSIGHT:', '', summary_match.group(1).strip()).strip()

        item_blocks = re.findall(r'\[(?:NEWS_ITEM|TECH_ITEM)\]\s*(.*?)(?=\s*\[|$)', text, re.DOTALL)
        items_lines = []
        outlook = outlook_match.group(1).strip() if outlook_match else ""

        def extract_field(pattern, block_text):
            # 尋找該欄位，直到遇到下一個欄位標籤或 block 結束
            match = re.search(pattern + r':\s*(.*?)(?=\s*(?:TITLE|URL|SUMMARY|INSIGHT|\[|$))', block_text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        if item_blocks:
            display_items = item_blocks[:5] # 只在 Discord 顯示 5 則
            for block in display_items:
                t_str = extract_field('TITLE', block)
                u_str = extract_field('URL', block)
                s_str = extract_field('SUMMARY', block)
                i_str = extract_field('INSIGHT', block)
                
                if t_str and u_str:
                    items_lines.append("----------------")
                    items_lines.append(f"**[{t_str}]({u_str})**")
                    if s_str: items_lines.append(f"• {s_str}")
                    if i_str: items_lines.append(f"> 💡 {i_str}")

        label = "🔮 **未來展望**" if "[FUTURE_OUTLOOK]" in text else "🕵️ **專家總結**"
        
        return {
            "summary": summary,
            "items_text": "\n".join(items_lines).strip(),
            "outlook": outlook,
            "outlook_label": label
        }
