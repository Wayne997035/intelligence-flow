import os
import sys
from notion_client import Client
from dotenv import load_dotenv
from datetime import datetime

# 將 src 加入路徑以便測試
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def test_notion_connection():
    load_dotenv()
    token = os.getenv("NOTION_INTEGRATION_SECRET")
    page_id = os.getenv("NOTION_DATABASE_ID")
    
    print(f"--- Notion 單元測試開始 ---")
    print(f"Token: {token[:10]}...")
    print(f"Target Page ID: {page_id}")
    
    if not token or not page_id:
        print("❌ 錯誤: .env 中缺少 NOTION_INTEGRATION_SECRET 或 NOTION_DATABASE_ID")
        return

    notion = Client(auth=token)
    
    # 1. 測試讀取頁面權限
    print("\n1. 正在嘗試讀取目標頁面資訊...")
    try:
        page = notion.pages.retrieve(page_id=page_id)
        print(f"✅ 成功讀取頁面！標題範例: {page.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', '無標題')}")
    except Exception as e:
        print(f"❌ 讀取頁面失敗: {e}")
        print("\n💡 提示: 請確認該頁面右上角 [...] -> [Add connections] 已加入你的 Integration。")
        return

    # 2. 測試建立子頁面
    print("\n2. 正在嘗試建立測試子頁面...")
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_page = notion.pages.create(
            parent={"page_id": page_id},
            properties={
                "title": [{"text": {"content": f"🛠️ Intel-Flow 測試連線 - {now_str}"}}]
            },
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"text": {"content": "如果你看到這行字，代表 Notion 連線與權限已完全正常！"}}]
                    }
                }
            ]
        )
        print(f"✅ 成功建立子頁面！")
        print(f"🔗 頁面連結: {new_page['url']}")
    except Exception as e:
        print(f"❌ 建立子頁面失敗: {e}")

if __name__ == "__main__":
    test_notion_connection()
