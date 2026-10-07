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

要跑測試的話改裝 dev 依賴（含 `requirements.txt` 全部內容 + pytest）：

```bash
pip install -r requirements-dev.txt
```

建立本機環境檔（建議）：

```bash
touch .env.local
```

## 常用指令

先跑測試：

```bash
./venv/bin/python -m pytest
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
./venv/bin/python main.py --once --live-delivery
```

`main.py` 以環境設定為預設值：`ENABLE_AI_ANALYSIS=true` 會啟用 Gemini/Groq 摘要，`DRY_RUN=true` 會阻止 Discord/Notion 實際發送。CLI 旗標只用於本次執行覆寫，例如 `--live-delivery` 會把本次執行切到正式發送。

## 排程

GitHub Actions 透過 `.github/workflows/intel-flow-schedule.yml` 執行：

- Asia/Taipei 09:00 daily
- Asia/Taipei 18:00 daily

排程使用 GitHub Actions `schedule.timezone`，時區設定為 `Asia/Taipei`。

GitHub 會在 repo 60 天沒有活動時自動停用排程。workflow 每次執行都會先呼叫 API 重新 enable 自己（`Keep schedule alive` 步驟）來避免這件事；若仍被停用，到 Actions → `Intel Flow Schedule` 按 **Enable workflow**。

手動驗證：在 Actions 頁面對 `Intel Flow Schedule` 按 **Run workflow** 並勾選 `dry_run`，會用真實來源與 AI 分析跑一輪但不發送 Discord/Notion，結果 `data/latest_run.json` 以 artifact `latest-run` 上傳，可檢查 `meta.ai_pipeline.irrelevant_dropped_sample` 等欄位。

## 設定重點

環境變數來源優先順序：

1. Shell / CI
2. `ENV_FILE` 指定的檔案（預設 `.env.local`）

常用參數：

| 變數 | 用途 |
| --- | --- |
| `ENABLE_AI_ANALYSIS` | 是否啟用 AI 產生摘要/洞察 |
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `NVIDIA_API_KEY` | AI 分析供應商，依序 Gemini → Groq → NVIDIA（build.nvidia.com）失敗才換下一家，至少要有一個 |
| `AI_PROVIDER_ORDER` | AI 供應商嘗試順序，預設 `gemini,groq,nvidia` |
| `NVIDIA_ALLOW_PRODUCTION` | 預設 `false`。NVIDIA 免費 API 依 [NVIDIA API Trial Terms](https://assets.ngc.nvidia.com/products/api-catalog/legal/NVIDIA%20API%20Trial%20Terms%20of%20Service.pdf) 僅限試用 / 開發 / 評估，正式使用需另購訂閱；因此排程正式發送預設不用 NVIDIA，只在 dry run 與 AI health check（評估）使用。有正式授權才設 `true` |
| `AI_MODEL` / `GROQ_MODEL` / `NVIDIA_MODEL` | 預設 `auto`：向各家查這把 key 可用的模型，自動挑最新的通用模型（Gemini 依序嘗試最多 5 個、每個等 30 秒，遇到 503/逾時就換下一個）；填入模型 id 則固定使用。可設為 repo variables |
| `ENABLE_DISCORD_DELIVERY` | 是否發送到 Discord |
| `ENABLE_NOTION_DELIVERY` | 是否發送到 Notion |
| `DRY_RUN` | 是否阻止外部發送（預設 `true`） |
| `STOCK_NEWS_LOOKBACK_DAYS` | 股市新聞時間窗（預設 7） |
| `AI_NEWS_LOOKBACK_DAYS` | AI 新聞時間窗（預設 7） |
| `AI_HIGH_IMPACT_LOOKBACK_DAYS` | AI 高衝擊延伸內容時間窗（預設 30） |
| `ENABLE_HISTORY_DEDUP` | 跨輪去重（預設關閉） |
| `NOTION_STATE_DB_ID` / `NOTION_STATE_DS_ID` | 跨輪去重狀態持久化用的 Notion database / data source id |
| `US_STOCKS` / `TW_STOCKS` | 追蹤標的（美股預設含 `SPCX` = SpaceX） |
| `STOCK_NAME_ALIASES` | 代號 → 公司/產品名，用於新聞搜尋與排序，格式 `SPCX:SpaceX\|Starlink,NVDA:Nvidia` |
| `STOCK_WATCH_TOPICS` / `AI_WATCH_TOPICS` | 透過 Google News RSS（免 key）額外追蹤的主題 |
| `AI_MODEL_WATCH` | 目前旗艦模型名稱（新聞精準搜尋用），新世代發表時更新即可，例如 `Claude Opus,GPT-6,GPT-6 Astra` |
| `ENABLE_GOOGLE_NEWS` / `ENABLE_TICKER_NEWS` | 開關 Google News 主題 / Yahoo Finance 個股新聞（預設開） |
| `TW_STOCK_SOURCE_ORDER` | 台股來源順序（預設 `yfinance,mis`） |

## 股市情報來源

- 報價：yfinance 取近一個月日線，計算日 / 5 日 / 1 月漲跌幅、1 月區間、量比（今日量 ÷ 近月均量）
- 新聞：NewsAPI（代號 + 公司別名）+ Yahoo Finance 個股新聞 + Google News 主題
- 排序後依標的 round-robin 交錯，避免清單前面的標的（例如 NVDA）把後面的標的（例如 SPCX）擠出報告

## AI 情報策略（摘要）

- 優先關注主流模型與平台脈絡：Claude / Gemini / xAI / Grok / Codex / ChatGPT
- 同時追蹤「新功能演進」：agent、skill、workflow、tooling 類訊號
- 重大 AI 資安事件（例如 frontier model、zero-day、breach、Mythos / Glasswing 類事件）可在主報時間窗外保留到 Notion 延伸內容
- Discord 與 Notion 各自會做內容去重；Notion 仍會包含 Discord 主報項目，但不會在 Notion 內重複渲染同一事件
- 社群熱度不是主排序，但會保留近期高討論度與 GitHub 趨勢專案

## 輸出檔案

- `data/latest_run.json`：本輪執行結果（payload + meta）
- `data/run_state.json`：跨輪去重狀態的**本機檔案備份**（`.gitignore` 排除，不會進 repo）

## 跨輪去重狀態持久化

`ENABLE_HISTORY_DEDUP=true` 啟用後，`RunStateStore` 透過可替換的儲存後端（`src/utils/state_backends.py`）讀寫去重狀態：

- **本機 / 未設定 Notion state DB 時**：退回 `FileStateBackend`，寫入 `data/run_state.json`。這個檔案只活在單一 process 的檔案系統上。
- **設定 `NOTION_STATE_DB_ID` + `NOTION_STATE_DS_ID`（且 `NOTION_TOKEN` 存在）時**：改用 `NotionStateBackend`，把狀態存成 Notion 專用資料庫裡的一列（單一 code block，JSON 依 2000 字元切成多個 rich_text chunk）。

**為什麼需要這個**：GitHub Actions 排程（`intel-flow-schedule.yml`）每次都是全新的 `ubuntu-latest` 容器，且該 workflow 沒有對 `data/` 做任何 cache 或 artifact 步驟。也就是說，若只用 `FileStateBackend`，`data/run_state.json` **不會**在兩次排程之間存活 —— 即使打開 `ENABLE_HISTORY_DEDUP`，跨輪去重在 server 上實際上從未真正生效過，且不會有任何錯誤訊息。要讓跨輪去重在排程環境下真的運作，MUST 設定 `NOTION_STATE_DB_ID` / `NOTION_STATE_DS_ID`（workflow 已透過對應 secrets 傳入）。

Notion 讀寫失敗（掛掉、rate limit、token 過期、權限被移除）時會降級為空狀態並記 warning log，本輪報告照常產出與送達，不會中止流程；代價是那一輪不去重。
