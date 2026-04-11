import requests
import re
from src.config import Config
from src.utils.logger import logger

class DiscordSender:
    def send_stock_and_analysis(self, us_stocks, tw_stocks, analysis_text, notion_url):
        """發送報價 + 股票分析 (第一篇)"""
        if not Config.DISCORD_WEBHOOK_URL: return
        
        # 報價部分
        us_lines = "\n".join([f"{s['symbol']}\n現:{s['price']} | 變:{s['change']}\n區:{s['range']}" for s in us_stocks])
        tw_lines = "\n".join([f"{s['symbol']}\n現:{s['price']} | 變:{s['change']}\n區:{s['range']}" for s in tw_stocks])

        # 清洗 AI 分析文字
        clean_text = self._format_text_with_links(analysis_text)
        
        description = f"🇺🇸 **美股**\n{us_lines}\n\n----------------\n"
        description += f"🇹🇼 **台股**\n{tw_lines}\n\n"
        description += f"----------------\n{clean_text}\n"
        
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
        
        clean_text = self._format_text_with_links(ai_text)
        description = f"{clean_text}\n"
        
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

    def _format_text_with_links(self, text):
        """清洗 AI 標籤並轉化為 Markdown 連結，隱藏原始網址"""
        lines = text.split('\n')
        new_lines = []
        
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 檢查是否遇到新標籤，若是則先結算 current_item
            if line.startswith('[') and current_item.get('title') and current_item.get('url'):
                new_lines.append(f"----------------\n**[{current_item['title']}]({current_item['url']})**")
                if current_item.get('summary'): new_lines.append(f"• {current_item['summary']}")
                if current_item.get('insight'): new_lines.append(f"> 💡 {current_item['insight']}")
                current_item = {}

            if line.startswith('[SECTION_SUMMARY]'):
                new_lines.append("📌 **摘要**")
            elif line.startswith('[EXPERT_VIEW]'):
                new_lines.append("\n🕵️ **專家總結**")
            elif line.startswith('[FUTURE_OUTLOOK]'):
                new_lines.append("\n🔮 **未來展望**")
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
                new_lines.append(line)

        # 處理最後一個 item
        if current_item.get('title') and current_item.get('url'):
            new_lines.append(f"----------------\n**[{current_item['title']}]({current_item['url']})**")
            if current_item.get('summary'): new_lines.append(f"• {current_item['summary']}")
            if current_item.get('insight'): new_lines.append(f"> 💡 {current_item['insight']}")

        final_text = "\n".join(new_lines)
        # 清理多餘空行
        final_text = re.sub(r"\n{3,}", "\n\n", final_text)
        return final_text.strip()
