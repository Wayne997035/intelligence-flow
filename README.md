# Intel-Flow

每天產出兩份報告：投資情報（股價 + 市場新聞）與 AI 技術情報，發送到 Discord（摘要）與 Notion（完整內容）。

## 執行

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

./venv/bin/python -m pytest                          # 測試
./venv/bin/python main.py --once --use-fixture       # 本地樣本資料
./venv/bin/python main.py --once                     # 真實來源，不發送
./venv/bin/python main.py --once --live-delivery     # 正式發送
```

排程：GitHub Actions `intel-flow-schedule.yml`，每天台北時間 09:00 / 18:00。

## 設定參數

GitHub：Settings → Secrets and variables → Actions。金鑰放 **Secrets**，其餘放 **Variables**；沒設就用 `src/config.py` 的預設值。本機用 `.env.local`。

**Secrets**

| 名稱 | 用途 |
| --- | --- |
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `NVIDIA_API_KEY` | AI 分析 |
| `NEWS_API_KEY` | NewsAPI |
| `DISCORD_WEBHOOK_URL` | Discord 發送 |
| `NOTION_INTEGRATION_SECRET` / `NOTION_DATABASE_ID` | Notion 報告 |
| `NOTION_STATE_DB_ID` / `NOTION_STATE_DS_ID` | 跨輪去重紀錄 |

**Variables**

| 名稱 | 用途 |
| --- | --- |
| `US_STOCKS` / `TW_STOCKS` | 美股 / 台股觀察清單（逗號分隔） |
| `STOCK_NAME_ALIASES` | 代號對應公司名，新聞搜尋用（`代號:名稱1\|名稱2,...`） |
| `STOCK_WATCH_TOPICS` / `AI_WATCH_TOPICS` | 額外追蹤的新聞主題 |
| `AI_MODEL_WATCH` | 旗艦模型名稱，新聞搜尋用 |
| `AI_GITHUB_RELEASE_REPOS` | 追蹤 release 的 GitHub repo |
| `STOCK_NEWS_LOOKBACK_DAYS` / `AI_NEWS_LOOKBACK_DAYS` | 新聞時間窗（天） |
| `AI_HIGH_IMPACT_LOOKBACK_DAYS` | 重大 AI 事件保留天數 |
| `HISTORY_TTL_HOURS` | 跨輪去重記憶時數 |
| `ENABLE_GOOGLE_NEWS` / `ENABLE_TICKER_NEWS` | 開關 Google News / Yahoo 個股新聞 |
| `AI_PROVIDER_ORDER` | AI 供應商嘗試順序 |
| `AI_MODEL` / `GROQ_MODEL` / `NVIDIA_MODEL` | 固定模型；預設 `auto` 自動挑選 |
| `NVIDIA_ALLOW_PRODUCTION` | 排程是否使用 NVIDIA |
