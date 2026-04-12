# Intel-Flow

Intel-Flow 用來整理兩條情報流：

1. 股市價格與相關新聞
2. AI 模型、研究、GitHub 專案、社群動態

輸出分成兩層：

- Discord：短摘要，適合快速看重點
- Notion：長內容，保留較完整的分析與回鏈

目前專案已改成「預設安全模式」：

- 不會預設呼叫 Gemini / Groq
- 不會預設發送到 Discord / Notion
- 可以先用 fixture 跑完整流程
- 會做同輪與跨輪去重，降低重複消息洗版

## Project Layout

```text
intelligence-flow/
├── main.py
├── src/
│   ├── ai/
│   ├── collectors/
│   ├── deliverers/
│   ├── models.py
│   ├── pipeline.py
│   └── utils/
├── tests/
│   └── fixtures/
└── docs/
```

主要模組：

- `main.py`
  負責收集資料、排序去重、分析、交付與 artifact 輸出
- `src/config.py`
  管理安全旗標、來源金鑰、去重與 artifact 路徑
- `src/pipeline.py`
  處理 canonical URL、標題正規化、排序與同輪去重
- `src/utils/state_store.py`
  處理跨輪去重，記住已送過的消息 fingerprint
- `src/ai/analyzer.py`
  預設用本地 synthesis；只有顯式開啟時才打 Gemini / Groq
- `src/deliverers/discord_sender.py`
  建 Discord embed payload
- `src/deliverers/notion_sender.py`
  建 Notion block payload

## Safety Defaults

這包最重要的原則是：先驗證流程，再碰真實外部服務。

預設值如下：

- `DRY_RUN=true`
- `ENABLE_AI_ANALYSIS=false`
- `ENABLE_DISCORD_DELIVERY=false`
- `ENABLE_NOTION_DELIVERY=false`
- `ENABLE_HISTORY_DEDUP=false`
- `WRITE_ARTIFACTS=true`
- `STOCK_NEWS_LOOKBACK_DAYS=7`
- `AI_NEWS_LOOKBACK_DAYS=14`

因此直接執行 `main.py`：

- 不會消耗 Gemini / Groq token
- 不會真的送 Discord webhook
- 不會真的寫 Notion
- 會把本輪結果寫到 `data/latest_run.json`

## Environment Variables

請參考 [`.env.example`](./.env.example)。

常用欄位：

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Gemini API key |
| `GROQ_API_KEY` | Groq API key |
| `AI_MODEL` | 預設 Gemini model |
| `NEWS_API_KEY` | NewsAPI key |
| `GITHUB_TOKEN` | GitHub API token |
| `DISCORD_WEBHOOK_URL` | Discord webhook |
| `NOTION_TOKEN` | Notion integration secret |
| `NOTION_PAGE_ID` | Notion database/page id |
| `DRY_RUN` | 是否禁止真實交付 |
| `ENABLE_AI_ANALYSIS` | 是否啟用 Gemini / Groq 分析 |
| `ENABLE_DISCORD_DELIVERY` | 是否啟用 Discord 發送 |
| `ENABLE_NOTION_DELIVERY` | 是否啟用 Notion 發送 |
| `ENABLE_HISTORY_DEDUP` | 是否啟用跨輪去重 |
| `STOCK_NEWS_LOOKBACK_DAYS` | 股市新聞時間窗（天） |
| `AI_NEWS_LOOKBACK_DAYS` | AI 新聞時間窗（天） |
| `HISTORY_TTL_HOURS` | 跨輪去重有效期，預設 `12` 小時 |
| `STATE_FILE` | 跨輪去重 state 檔路徑 |
| `ARTIFACT_FILE` | 本輪輸出 artifact 路徑 |
| `TW_STOCK_SOURCE_ORDER` | 台股來源順序，預設 `yfinance,mis` |

環境載入優先順序：

1. shell / CI 已存在的環境變數
2. `.env.local`
3. `.env`

## Install

建議使用獨立 `venv`：

```bash
cd intelligence-flow
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Local Verification

先跑測試：

```bash
cd intelligence-flow
python3 -m unittest discover -s tests
```

再跑完整 dry-run：

```bash
cd intelligence-flow
python3 main.py --once --use-fixture
```

這會：

- 讀取 `tests/fixtures/sample_bundle.json`
- 走完整 pipeline
- 產生 `data/latest_run.json`
- 不打真實 AI / Discord / Notion

## Real Execution

只有在你確認測試與 dry-run 都正常後，才建議啟用真實交付：

```bash
cd intelligence-flow
ENABLE_AI_ANALYSIS=true \
ENABLE_DISCORD_DELIVERY=true \
ENABLE_NOTION_DELIVERY=true \
python3 main.py --once --live-delivery --enable-ai
```

live 模式啟動前會先做 preflight：

- 開啟 AI 分析但沒有 Gemini/Groq key 時直接報錯
- 開啟 Discord live delivery 但 webhook 缺失時直接報錯
- 開啟 Notion live delivery 但 token/page id 缺失時直接報錯

排程模式：

```bash
cd intelligence-flow
python3 main.py --schedule
```

目前排程是每天 `08:00` 與 `20:00`。

## Dedup Strategy

這包現在有兩層去重：

### 1. Same-run dedup

在同一輪收集裡，會依據：

- canonical URL
- normalized title

合併同篇或鏡像文章，避免同則新聞因轉載重複出現。

### 2. Cross-run dedup

已送過的消息 fingerprint 會記到 `data/run_state.json`。下一輪若再次抓到同消息，會在 `HISTORY_TTL_HOURS` 有效期內優先略過；預設是 `12` 小時，避免短時間重複洗版，但不會永久擋住舊題材。

### 3. Source-quality preference

當不同來源其實在講同一則消息時，系統會優先保留來源品質較高的版本。

目前大致排序：

1. `official_news`
2. `model_release`
3. `research`
4. `github_repo`
5. `news`
6. `community`

這代表：

- 官方公告會優先於 Reddit/社群轉述
- 論文與模型發布會優先於二手整理文
- 同則 AI 或股票相關消息不會只因為社群文更長就蓋掉官方來源

## Artifacts

預設會輸出：

- `data/latest_run.json`
  本輪執行後的結構化結果
- `data/run_state.json`
  已見過消息的去重 state，含 `seen_at` 時間戳，預設只保留 12 小時內有效紀錄

這兩個檔都已被 `.gitignore` 忽略。

## Source Notes

目前來源包含：

- 股票：TWSE、yfinance
- 台股來源順序可設定：`TW_STOCK_SOURCE_ORDER=yfinance,mis`
- 一般新聞：NewsAPI
- 官方 AI 更新：OpenAI News、Anthropic News、Google Developers Blog、GitHub Blog feed
- Curated release watcher：GitHub Releases watchlist
- AI 社群/專案：Hacker News、GitHub、Reddit
- 模型/論文：Hugging Face、arXiv

注意：

- `yfinance` 適合原型與監控，不是交易級資料源
- `mis.twse` 可以保留，但應搭配 fallback，不要讓單一來源失敗拖垮整體流程
- NewsAPI 對「最新」消息不一定是最佳方案
- Reddit / 社群資料應視為輔助訊號，不該和官方公告同權重
- curated GitHub release watchlist 可透過 `AI_GITHUB_RELEASE_REPOS` 調整

## Tests

目前測試涵蓋：

- pipeline 去重與 URL 正規化
- analyzer fallback 結構
- Discord payload 生成
- Notion block 生成
- main fixture dry-run
- state store 跨輪去重

## Repo Hygiene

這個 repo 已經修正 `.gitignore`，並將：

- `.env`
- 意外的 secret dump 檔
- `__pycache__`
- `.pyc`
- `.DS_Store`

從 Git tracking 中移除。

但如果你之前的 key 已經進過 Git 歷史，仍然建議立刻 rotate。

## Review

完整 review 與 roadmap 參考：

- [`docs/review-and-roadmap.md`](./docs/review-and-roadmap.md)
- [`CODEX.md`](./CODEX.md)
