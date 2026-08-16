<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — 降低 AI 运营成本的即插即用上下文保障</h1>

<p align="center"><b>在不失去对关键证据控制的前提下减少不必要的上下文。</b><br>
优先选择最高价值的证据，压缩它，保持原始数据可恢复，并发出回执 — 无需重写代码库或代理架构。</p>
<p align="center">
  <sub>Entroly 是一个本地优先的 Context OS：内容寻址的证据、可恢复的压缩和可审计的回执。通过代理、MCP、插件、包装器和 SDK 路径与 Claude Code、Codex、OpenClaw、GitHub Copilot、Cursor、Aider 以及 OpenAI/Anthropic 兼容应用协同工作。</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438 次下载 · 持续增长中</b><br>
<sub>统计自不同分发源。</sub></p>

<p align="center"><b>⭐ 如果 Entroly 对你有用，请在 GitHub 上给项目加星。</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ 在 GitHub 上给 Entroly 加星</a> — 这有助于项目成长并被更多开发者发现。</p>

<p align="center">
  <b><a href="../../README.md">English</a> · 简体中文 · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Entroly 是什么？（通俗解释）

AI 编程助手有记忆限制。把整个代码库交给它，它会变慢、变贵、容易分心 — 就像给一个人500页的手册，而他只需要第47页。

**Entroly 帮你找到第47页。**

它位于你的代码和 AI 之间，读取所有内容，只传递与当前问题相关的部分。三件事使这样做是安全的：
|  |  |
|---|---|
| 💰 **你的账单会降低** | 发送给 AI 的内容更少意味着更小的账单。具体节省多少取决于工作内容 — 参见下方的[实际数据](#基准测试)。 |
| 🔍 **不会丢失任何内容** | Entroly 搁置的所有内容都会保留，可以*完全*恢复，逐字逐句。 |
| 🧾 **你可以检查它的工作** | 每个决策都附带回执：保留了什么，遗漏了什么，以及原因。 |

**我需要修改代码吗？** 不需要。Entroly 与你已经使用的工具配合工作 — Claude Code、Cursor、Copilot 以及30多种其他工具 — 在后台运行。

**试用需要付费吗？** 不需要。下面[安装](#安装)部分的两个命令完全在你的本地机器上运行，无需 API 密钥，在你连接任何付费服务之前就能展示你自己项目的真实数据。

---
## 安装

> **不确定选哪个？** 选择 **Python**。它是最完整的版本，也是大多数人使用的。其他选项是运行同一引擎的替代方式。

| 平台 | 安装命令 | 获得的功能 |
|---|---|---|
| 🐍 **Python** (pip) — *推荐* | `pip install -U entroly` | 全部功能：命令行工具、AI 编辑器通信服务器和代码库 |
| 📦 **Node / npm** | `npm install -g entroly` | 同一引擎，无需 Python |
| 🦀 **Rust**（源码构建） | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | 一个独立的程序，无需 Python 或 Node |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | macOS/Linux 上的命令行工具 |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | 在容器中运行，无需在机器上安装任何东西 |

**现在检查安装是否成功 — 免费、离线、无需 API 密钥：**

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

<sub>两个命令都在本地运行。都不会调用 AI 或产生任何费用。</sub>

---
## 快速开始 — 按你的工作方式

> **只想让它工作？** `pip install -U entroly && entroly go` — 就这么简单。它会找到你的编辑器，自动配置，并显示前后对比的仪表板。

| 你的情况 | 这样做 | 获得的效果 |
|---|---|---|
| 🟢 **"我只想让它运行。"** *(pip / Python 用户)* | `pip install -U entroly && entroly go` | 自动检测你的编辑器，包装你的代理，打开显示令牌前后对比的仪表板 |
| **"我用 Node，不用 Python。"** *(npm 用户)* | `npm install -g entroly && entroly init` | 同一引擎，无需 Python |
| **"我用 Claude Code / Cursor / Windsurf / VS Code。"** *(MCP 用户)* | `entroly attach create --client claude --project . --ttl 4h --install` | 你的编辑器获得压缩、回执和恢复功能作为内置工具 |
| **"我在用 Python 构建自己的应用。"** *(SDK 用户)* | `from entroly import compress, compress_messages, optimize` | 直接在代码中调用，随时随地组装提示 |
| **"我有 API 密钥和自己的应用。"** *(代理用户)* | `entroly proxy` → 将 `OPENAI_BASE_URL` 指向 `localhost:9377` | 每个请求在经过时都会被优化 — 你这边无需任何代码更改 |

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="修复登录 bug")
```

---
## 基准测试

重要的问题是：**如果发送更少的内容，AI 会开始出错吗？** 这些是标准的公开测试，分别在使用和不使用 Entroly 的情况下运行。

| 基准测试 | 基线 | 使用 Entroly | 保留率 | 令牌节省 |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |
| GSM8K | 85% | 85% | **100%** | 直通* |

<sub>*直通：上下文已经在预算内，保持不变。每行 n=20-50。</sub>

**坦率地说：** 看 SQuAD 2.0 行 — 准确率*下降*了（80% → 72%）。压缩是一种权衡，不是魔法，它不是在所有地方都能赢。这就是 `entroly simulate` 存在的原因：在你的项目上运行它，在承诺任何事情之前看看你自己的数据。

---
## 功能特性

- **先选择，再压缩** — 先确定哪些文件真正回答了你的问题，*然后*再压缩它们。
- **精确返回原始内容** — 任何被搁置的内容都可以逐字逐句地恢复，并通过指纹验证。
- **展示工作过程** — 每个决策的回执：保留了什么，遗漏了什么和原因，以及剩余的风险。
- **事实核查答案** — 将 AI 的回答与提供给它的证据进行比较，在你的机器上完成，无需为第二次 AI 调用付费。
- **不破坏你的缓存** — 保持提示中不变的部分稳定，使你的提供商对重复文本的折扣仍然适用。

---
## 常见问题

<details>
<summary><b>这会改变我的代码或文件吗？</b></summary>
<br>
不会。Entroly 读取你的文件并决定将什么发送给 AI。它永远不会编辑、移动或删除你项目中的任何内容。
</details>

<details>
<summary><b>我的代码会被上传到任何地方吗？</b></summary>
<br>
不会。所有的选择、压缩和检查都在你自己的机器上进行。Entroly 不会发出任何自己的外部调用 — 离开你计算机的唯一内容是你已经发送给 AI 提供商的请求，只是更小了。默认不开启任何分析。
</details>

<details>
<summary><b>如果它遗漏了重要内容怎么办？</b></summary>
<br>
没有任何内容被丢弃。任何被遗漏的内容都会被存储，可以完全恢复 — `entroly recover` 会逐字逐句地返回原始内容，并通过指纹验证。
</details>

<details>
<summary><b>这实际上能为我省多少钱？</b></summary>
<br>
说实话：这取决于你的项目。在你的项目中运行 `entroly simulate` — 它是免费的，不需要 API 密钥，可以估算你自己文件的缩减量。如果你的提示已经很小，Entroly 会原样传递。
</details>

---
## 文档和社区

- **[完整基准测试证据](../../docs/BENCHMARKS.md)** — 所有数据、协议、工件和注意事项。
- **[产品功能图](../../docs/product-surface.md)** — CLI、SDK、MCP、代理、验证、记忆、安全。
- **[架构和完整规范](../../docs/DETAILS.md)** — Rust 模块、压缩、溯源、命令参考。
- **[代理兼容性](../../docs/agent-compatibility.md)** — 每个支持的客户端及其确切的认证边界。
- **[局限性](../../docs/limitations.md)** — Entroly 在哪些方面有帮助，在哪些方面直通，不保证什么。
- **[Cookbook](../../cookbook/README.md)** — 即用配方。
- **[Discord](https://juyterman1000.github.io/entroly/docs/discord.html)** · **[Discussions](https://github.com/juyterman1000/entroly/discussions)** · **[Issues](https://github.com/juyterman1000/entroly/issues)**

<p align="center"><sub>Apache-2.0 · 本地优先 · 默认不发送外部分析数据</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
