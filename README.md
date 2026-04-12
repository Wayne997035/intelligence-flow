# Intel-Flow

Intel-Flow 是一個情報整理器，固定產出兩份報告：

- 投資情報（股價 + 市場新聞）
- AI 技術情報（官方發布 + GitHub + 社群 + 研究）

輸出渠道：

- Discord：短摘要，適合快速瀏覽
- Notion：完整內容，保留來源與回鏈

## 核心原則

- Recency first：預設只看最近時間窗（股市 7 天、AI 7 天）
- Source quality first：官方發布、release、研究優先於一般社群噪音
- Safe by default：預設 dry-run，不會直接發送到外部服務

## 快速開始

```bash
cd intelligence-flow
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

建立本機環境檔（建議）：

```bash
touch .env.local
```

## 常用指令

先跑測試：

```bash
./venv/bin/python -m unittest discover -s tests
```

本地 fixture 驗證（不打外部來源）：

```bash
./venv/bin/python main.py --once --use-fixture
```

真實來源 dry-run（不發送 Discord/Notion）：

```bash
./venv/bin/python main.py --once
```

正式發送（Discord + Notion）：

```bash
./venv/bin/python main.py --once --live-delivery --enable-ai
```

## 設定重點

環境變數來源優先順序：

1. Shell / CI
2. `ENV_FILE` 指定的檔案（預設 `.env.local`）

常用參數：

| 變數 | 用途 |
| --- | --- |
| `ENABLE_AI_ANALYSIS` | 是否啟用 AI 產生摘要/洞察 |
| `ENABLE_DISCORD_DELIVERY` | 是否發送到 Discord |
| `ENABLE_NOTION_DELIVERY` | 是否發送到 Notion |
| `STOCK_NEWS_LOOKBACK_DAYS` | 股市新聞時間窗（預設 7） |
| `AI_NEWS_LOOKBACK_DAYS` | AI 新聞時間窗（預設 7） |
| `ENABLE_HISTORY_DEDUP` | 跨輪去重（預設關閉） |
| `US_STOCKS` / `TW_STOCKS` | 追蹤標的 |
| `TW_STOCK_SOURCE_ORDER` | 台股來源順序（預設 `yfinance,mis`） |

## AI 情報策略（摘要）

- 優先關注主流模型與平台脈絡：Claude / Gemini / xAI / Grok / Codex / ChatGPT
- 同時追蹤「新功能演進」：agent、skill、workflow、tooling 類訊號
- 社群熱度不是主排序，但會保留近期高討論度與 GitHub 趨勢專案

## 輸出檔案

- `data/latest_run.json`：本輪執行結果（payload + meta）
- `data/run_state.json`：跨輪去重狀態（啟用時才使用）

## 安全與版控

- 不要提交任何 `.env` 類檔案或金鑰
- `.gitignore` 已忽略 `.env`, `.env.local`, `.env.*`
- 若曾提交過機敏資料，請先輪替金鑰再推送遠端
