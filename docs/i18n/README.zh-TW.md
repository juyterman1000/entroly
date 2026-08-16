<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — 降低 AI 營運成本的即插即用上下文保障</h1>

<p align="center"><b>在不失去對關鍵證據控制的前提下減少不必要的上下文。</b><br>
優先選擇最高價值的證據，壓縮它，保持原始資料可復原，並發出收據 — 無需重寫代碼庫或代理架構。</p>
<p align="center">
  <sub>Entroly 是一個本地優先的 Context OS：內容定址的證據、可復原的壓縮和可審計的收據。透過代理、MCP、外掛程式、包裝器和 SDK 路徑與 Claude Code、Codex、OpenClaw、GitHub Copilot、Cursor、Aider 以及 OpenAI/Anthropic 相容應用協同運作。</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438 次下載 · 持續增長中</b><br>
<sub>統計自不同分發源。</sub></p>

<p align="center"><b>⭐ 如果 Entroly 對你有幫助，請在 GitHub 上給專案加星。</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ 在 GitHub 上給 Entroly 加星</a> — 這有助於專案成長並讓更多開發者發現。</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · 繁體中文 · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Entroly 是什麼？（通俗解釋）

AI 程式設計助手有記憶限制。把整個代碼庫交給它，它會變慢、變貴、容易分心 — 就像給一個人 500 頁的手冊，而他只需要第 47 頁。

**Entroly 幫你找到第 47 頁。**

它位於你的代碼和 AI 之間，讀取所有內容，只傳遞與當前問題相關的部分。三件事使這樣做是安全的：
|  |  |
|---|---|
| 💰 **你的帳單會降低** | 發送給 AI 的內容更少意味著更小的帳單。具體節省多少取決於工作內容 — 參見下方的[基準測試](#基準測試)。 |
| 🔍 **不會遺失任何內容** | Entroly 擱置的所有內容都會保留，可以*完全*復原，一字不差。 |
| 🧾 **你可以檢查它的工作** | 每個決策都附帶收據：保留了什麼，遺漏了什麼，以及原因。 |

**我需要修改代碼嗎？** 不需要。Entroly 與你已經使用的工具配合運作 — Claude Code、Cursor、Copilot 以及 30 多種其他工具 — 在背景運行。

**試用需要付費嗎？** 不需要。下面[安裝](#安裝)部分的兩個命令完全在你的本機上運行，無需 API 金鑰，在你連接任何付費服務之前就能展示你自己專案的真實數據。

---
## 安裝

> **不確定選哪個？** 選擇 **Python**。它是最完整的版本，也是大多數人使用的。其他選項是運行同一引擎的替代方式。

| 平台 | 安裝命令 | 獲得的功能 |
|---|---|---|
| 🐍 **Python** (pip) — *推薦* | `pip install -U entroly` | 全部功能：命令列工具、AI 編輯器通訊伺服器和代碼庫 |
| 📦 **Node / npm** | `npm install -g entroly` | 同一引擎，無需 Python |
| 🦀 **Rust**（原始碼建置） | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | 一個獨立的程式，無需 Python 或 Node |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | macOS/Linux 上的命令列工具 |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | 在容器中運行，無需在機器上安裝任何東西 |

**現在檢查安裝是否成功 — 免費、離線、無需 API 金鑰：**

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## 快速開始 — 按你的工作方式

> **只想讓它運作？** `pip install -U entroly && entroly go` — 就這麼簡單。它會找到你的編輯器，自動設定，並顯示前後對比的儀表板。

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="修復登入 bug")
```

---
## 基準測試

| 基準測試 | 基準線 | 使用 Entroly | 保留率 | Token 節省 |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## 功能特性

- **先選擇，再壓縮** — 先確定哪些檔案真正回答了你的問題，*然後*再壓縮它們。
- **精確復原原始內容** — 任何被擱置的內容都可以逐字逐句地復原，並透過指紋驗證。
- **展示工作過程** — 每個決策的收據：保留了什麼，遺漏了什麼和原因，以及剩餘的風險。
- **事實查核答案** — 將 AI 的回答與提供給它的證據進行比較，在你的機器上完成，無需為第二次 AI 呼叫付費。
- **不破壞你的快取** — 保持提示中不變的部分穩定，使你的提供商對重複文字的折扣仍然適用。

---
## 常見問題

<details>
<summary><b>這會改變我的代碼或檔案嗎？</b></summary>
<br>
不會。Entroly 讀取你的檔案並決定將什麼發送給 AI。它永遠不會編輯、移動或刪除你專案中的任何內容。
</details>

<details>
<summary><b>我的代碼會被上傳到任何地方嗎？</b></summary>
<br>
不會。所有的選擇、壓縮和檢查都在你自己的機器上進行。預設不開啟任何分析。
</details>

---

<p align="center"><sub>Apache-2.0 · 本地優先 · 預設不發送外部分析數據</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
