<p align="center">
  <a href="../../README.md">🇬🇧 English</a> •
  <a href="README.ja.md">🇯🇵 日本語</a> •
  <a href="README.ko.md">🇰🇷 한국어</a> •
  <a href="README.pt-BR.md">🇧🇷 Português</a> •
  <a href="README.es.md">🇪🇸 Español</a> •
  <a href="README.de.md">🇩🇪 Deutsch</a> •
  <a href="README.fr.md">🇫🇷 Français</a> •
  <a href="README.ru.md">🇷🇺 Русский</a> •
  <a href="README.hi.md">🇮🇳 हिन्दी</a> •
  <a href="README.tr.md">🇹🇷 Türkçe</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Token_Savings-up_to_95%25-brightgreen?style=for-the-badge" alt="Token节省: 高达95%">
  <img src="https://img.shields.io/badge/Learning_Cost-$0-blue?style=for-the-badge" alt="学习成本: $0">
  <img src="https://img.shields.io/badge/Engine-Rust_%2B_WASM-orange?style=for-the-badge&logo=rust" alt="Rust + WASM">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <a href="https://github.com/juyterman1000/entroly-cost-check-"><img src="https://img.shields.io/badge/GitHub_Action-Cost_Check-purple?style=for-the-badge&logo=githubactions" alt="GitHub Action"></a>
  <a href="https://mcpmarket.com/daily/top-mcp-server-list-march-26-2026"><img src="https://img.shields.io/badge/%231_MCP_Market-Ranked_Server-gold?style=for-the-badge&logo=starship&logoColor=white" alt="MCP Market 第一名"></a>
</p>

<h1 align="center">Entroly — 将AI Token成本降低70–95%</h1>

<h3 align="center">你的AI编程工具只能看到代码库的5%。<br/>Entroly让它们看到全貌——只需极低成本。</h3>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>在线演示 →</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI">
  <img src="https://img.shields.io/npm/v/entroly?color=red&label=npm">
  <img src="https://img.shields.io/badge/Tests-484_passing-success">
  <img src="https://img.shields.io/badge/Latency-<10ms-purple">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green">
</p>

---

## 问题——以及对底线的影响

每个AI编程工具——Claude、Cursor、Copilot、Codex——都有同一个盲点：**它一次只能看到5-10个文件。** 其余95%的代码库完全不可见。这导致API幻觉、broken imports、遗漏依赖，以及开发者数小时修复AI生成的错误。

模型不断变大——**Claude Opus 4.7** 刚刚发布，功能更强，但每个token的成本也更高。更大的上下文窗口并不能解决问题；它们只会让问题更糟。你每次请求都在为186,000个token买单——其中大部分是重复的样板代码。

> **Entroly在30秒内解决这两个问题。** 它将你的整个代码库以可变分辨率压缩进AI上下文窗口，让你的AI看到一切——而你几乎不需要为此付费。

---

## 第一天就改变什么

| 指标 | 使用Entroly之前 | **使用Entroly之后** |
|---|---|---|
| AI可见的文件 | 5–10个 | **你的整个代码库** |
| 每次请求Token数 | ~186,000 | **9,300 – 55,000** |
| 月度AI支出(1K请求/天) | ~$16,800 | **$840 – $5,040** |
| AI回答准确率 | 不完整，经常幻觉 | **依赖感知，准确** |
| 开发者修复AI错误的时间 | 每周数小时 | **几乎为零** |
| 设置 | 数天的Prompt工程 | **30秒** |

> **ROI示例：** 一个10人团队每月AI API支出$15K，第一天就能节省**$10K–$14K/月**。Entroly在第一个小时就能回本。（它是免费开源的，所以其实是即时回本。）

---

## 你的竞争对手已经知道的

今天采用Entroly的团队不只是在省钱——他们正在**累积你的团队无法追赶的优势**。

- **第1周：** 他们的AI看到100%的代码库。你的只看到5%。他们交付更快。
- **第1个月：** 他们的运行时已经学会了代码库模式。你的还在幻觉import。
- **第3个月：** 他们的安装已接入联邦——从全球数千个团队吸收优化策略。你根本不知道这东西的存在。
- **第6个月：** 他们已经节省了$80K+的API成本。预算投在了招聘上。你还在向财务解释为什么AI账单一直在涨。

每等一天，差距就更大。联邦效应意味着**早期采用者变强更快**——这个优势会复利增长。

---

## 原理（30秒）

```bash
npm install entroly-wasm && npx entroly-wasm
# 或
pip install entroly && entroly go
```

就这样。Entroly自动检测你的IDE，连接Claude/Cursor/Copilot/Codex/MiniMax，开始优化。无需配置。无需YAML。无需embeddings。

**内部发生了什么：**

1. **索引** — 2秒内映射你的整个代码库
2. **评分** — 按信息密度对每个文件排名
3. **选择** — 为你的token预算选择数学最优子集
4. **交付** — 关键文件完整传输，支持文件作为签名，其余作为引用
5. **学习** — 跟踪有效方案，随时间变得更智能

你的AI现在看到100%的代码库。你只需支付5-30%的token。

---

## 竞争优势——Entroly的不同之处

### 🧠 无需额外成本就能变得更智能

大多数"自我改进"AI工具靠消耗token来学习——你的账单随着它们的智能增长。Entroly的学习循环是**可证明的token负增长**：它不可能在学习上花费超过为你节省的。

算法简单且可审计：

```
学习预算 ≤ 5% × 终身节省额
```

第1天：70%token节省。第30天：85%+。第90天：90%+。**改进成本$0。**

### 🌐 联邦群体学习——听起来像科幻小说的部分

现在把梦境循环乘以**地球上每一个运行Entroly的开发者。**

当你睡觉时，你的守护进程在做梦——其他10,000个也是。每一个都发现了略有不同的代码压缩技巧。每一个都匿名分享了它学到的。每一个都吸收了其他人发现的东西。

**你醒来。你的AI比你离开时更聪明了。不是因为你做了什么——而是因为群体所梦到的。**

```
你的守护进程做梦 → 发现更好的策略 → 匿名分享
     ↓
10,000个其他守护进程昨晚做了同样的事
     ↓
你打开笔记本电脑 → 你的AI已经吸收了所有这些
```


**网络效应：**
- 每个新用户让其他所有人的AI更好——这个安装基数无法被fork
- 你的代码永远不会移动。只有优化权重——带噪声保护且匿名
- 基础设施成本：**$0**。运行在GitHub上。没有服务器。没有GPU。没有云端

```bash
# 选择加入——永远是你的选择
export ENTROLY_FEDERATION=1
```

### ✂️ 响应蒸馏——输出也节省Token

LLM响应包含约40%的填充内容——"当然！我很乐意帮忙！"、含糊其辞、元评论。Entroly把它们剥掉。代码块永远不会被修改。

```
之前: "当然！我很乐意帮忙。让我看看你的代码。
       问题出在auth模块。希望对你有帮助！"

之后: "问题出在auth模块。"
       → 减少70%输出token
```

三个强度等级：`lite` → `full` → `ultra`。一个环境变量即可启用。

### 🔒 本地运行。你的代码永远不会离开你的机器。

零云端依赖。零数据泄露风险。一切在你的CPU上运行，不到10ms。适用于物理隔离和受监管的环境——没有任何数据会外传。

---

## 适配你的技术栈

| 工具 | 设置 |
|---|---|
| **Cursor** | `entroly init` → MCP server |
| **Claude Code** | `claude mcp add entroly -- entroly` |
| **GitHub Copilot** | `entroly init` → MCP server |
| **Codex CLI** | `entroly wrap codex` |
| **Windsurf / Cline / Cody** | `entroly init` |
| **任意LLM API** | `entroly proxy` → HTTP代理 `localhost:9377` |

同样支持：OpenAI API • Anthropic API • LangChain • LlamaIndex • MCP-native

---

## 基准测试

### 实时进化追踪

这来自本仓库的vault，不是路线图：

```
[detect]     发现缺口 → entity="auth", miss_count=3
[synthesize] StructuralSynthesizer ($0, 确定性, 无LLM)
[benchmark]  skill=ddb2e2969bb0 → fitness 1.0 (1 pass / 0 fail, 338 ms)
[promote]    status: draft → promoted
[spend]      $0.0000 — 不变式 C_spent ≤ τ·S(t) 成立
```

### 准确率保持

压缩不影响准确率——我们测量过（n=100, gpt-4o-mini, Wilson 95% CI）：

| 基准测试 | 基线 (95% CI) | 使用Entroly (95% CI) | 保持率 |
|---|---|---|---|
| NeedleInAHaystack | 100% [83.9–100%] | 100% [83.9–100%] | **100.0%** |
| GSM8K | 85.0% [76.7–90.7%] | 86.0% [77.9–91.5%] | **101.2%** |
| SQuAD 2.0 | 84.0% [75.6–89.9%] | 83.0% [74.5–89.1%] | **98.8%** |
| MMLU | 82.0% [73.3–88.3%] | 85.0% [76.7–90.7%] | **103.7%** |
| TruthfulQA (MC1) | 72.0% [62.5–79.9%] | 73.0% [63.6–80.7%] | **101.4%** |
| LongBench (HotpotQA) | 57.0% [47.2–66.3%] | 59.8% [49.8–69.0%] | **104.9%** |

> 6项基准测试的置信区间全部重叠——准确率与基线在统计学上无法区分。LongBench（唯一上下文超出预算的基准）在节省3.6%token的同时，保持率反而**提升**。复现：`python -m bench.accuracy --benchmark all --model gpt-4o-mini --samples 100`

### CI/CD集成

在每个PR中运行token成本检查——在上线前捕获回退：

```yaml
- uses: juyterman1000/entroly-cost-check-@v1
```

→ **[entroly-cost-check GitHub Action](https://github.com/juyterman1000/entroly-cost-check-)**

---

## 实时观看运行——即时通知

三个聊天集成开箱即用。实时查看每次缺口检测、技能合成和梦境循环胜利：

```bash
export ENTROLY_TG_TOKEN=...          # Telegram（双向：/status /skills /gaps /dream）
export ENTROLY_DISCORD_WEBHOOK=...   # Discord
export ENTROLY_SLACK_WEBHOOK=...     # Slack
```

---

## 可移植技能（agentskills.io）

Entroly创建的技能不会被锁定。导出为开放的agentskills.io v0.1规范：

```bash
node node_modules/entroly-wasm/js/agentskills_export.js ./dist/agentskills
python -m entroly.integrations.agentskills ./dist/agentskills
```

每个导出的技能都带有 `origin.token_cost: 0.0`——零成本来源也跟随传递。

---

## 完全对等：Python & Node.js

两个运行时功能完整。相同引擎，相同vault，相同学习循环：

| 能力 | Python | Node.js (WASM) |
|---|---|---|
| 上下文压缩 | ✅ | ✅ |
| 自我进化 | ✅ | ✅ |
| 梦境循环 | ✅ | ✅ |
| 联邦 | ✅ | ✅ |
| 响应蒸馏 | ✅ | ✅ |
| 聊天网关 | ✅ | ✅ |
| agentskills.io导出 | ✅ | ✅ |

---

## 深入了解

架构、21个Rust模块、3分辨率压缩、来源保证、RAG对比、完整CLI参考、Python SDK、LangChain集成 → **[docs/DETAILS.md](../DETAILS.md)**

---

<p align="center">
  <b>别再为AI浪费的Token买单。开始运行一个能教自己的AI。</b><br/>
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>

<p align="center">
  <a href="https://github.com/juyterman1000/entroly/discussions">讨论</a> •
  <a href="https://github.com/juyterman1000/entroly/issues">问题</a> •
  MIT 许可证
</p>
