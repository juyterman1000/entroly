> **Archived translation:** This file is not aligned with the current evidence-backed README and may contain outdated or unsupported claims. Use the [canonical README](../../README.md) and [public evidence policy](../public-evidence.md).

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
  <img src="https://img.shields.io/badge/Token_Savings-workload_dependent-brightgreen?style=for-the-badge" alt="Token节省取决于工作负载">
  <img src="https://img.shields.io/badge/Local_First-no_embeddings_API-blue?style=for-the-badge" alt="本地优先">
  <img src="https://img.shields.io/badge/Engine-Rust_%2B_WASM-orange?style=for-the-badge&logo=rust" alt="Rust + WASM">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
</p>

<h1 align="center">在 AI 编码中将 Claude / OpenAI / Gemini 账单削减 70–95%。</h1>

<h3 align="center">压缩上下文，保持服务商缓存命中，并用 $0 幻觉防护验证每个回答。<br/>即插即用支持 Cursor、Claude Code、Codex、Aider 等 34+ 工具及自定义服务商 — 30 秒，无需改动代码。</h3>

<p align="center"><sub>可审计的上下文控制平面 · 每个回答都附带回执：使用了什么、省略了什么、为什么、以及还剩哪些风险 · 本地优先 · Rust + WASM · 可逆 · 节省量基于真实工作负载实测</sub></p>

<p align="center">
  <a href="../../README.md#install"><b>安装</b></a> ·
  <a href="../../cookbook/README.md"><b>Cookbook</b></a> ·
  <a href="../../README.md#benchmarks"><b>基准测试</b></a> ·
  <a href="../../README.md#works-with-your-stack"><b>21 支持的集成</b></a> ·
  <a href="https://juyterman1000.github.io/entroly/docs/dashboard.html"><b>仪表盘</b></a>
</p>

<p align="center">
  <code>npm install -g entroly && entroly</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>在线演示 →</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/self_improvement.svg" alt="Entroly 自我改进" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/token_savings.svg" alt="Entroly 利润" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/context_quality.svg" alt="Entroly 上下文质量" width="800">
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

发送原始文件转储的AI编程工具通常面临同一个限制：**模型一次可能只收到少量文件。** 代码库的其余部分未被表示。这可能导致不支持的API引用、遗漏依赖，以及开发者需要额外时间来验证AI生成的建议。

> Entroly通过从完整仓库中选择紧凑的、可变分辨率的上下文来解决这个问题。

---

## 第一天就改变什么

| 指标 | 未使用Entroly | **使用Entroly** |
|---|---|---|
| AI可见的文件 | 5–10个 | **支持的文件以可变分辨率选择** |
| 每次请求Token数 | ~186,000原始示例 | **发布检查中为9,300 – 55,000** |
| 月度AI支出 | 取决于供应商/模型 | **当输入token减少时降低** |
| AI回答可验证性 | 取决于提供的上下文 | **可对照选择的证据审计** |
| 设置 | 手动Prompt工程 | **30秒** |

> 节省取决于仓库大小、查询范围、模型定价和预算。在你自己的仓库上运行 `entroly demo` 或 `entroly verify-claims` 进行本地测量。

---


---

## 原理（30秒）

```bash
npm install -g entroly && entroly
# 或
pip install entroly && entroly go
```

就这样。Entroly自动检测你的IDE，连接支持的编程工具，开始优化。无需配置。无需YAML。无需embeddings。

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

Entroly的大部分排名和反馈循环在本地运行，不需要嵌入API或模型调用。

当启用可选的合成或联网学习时，预算门限旨在控制成本：

```
学习预算目标 ≤ 5% × 终身节省额
```

默认情况下，本地上下文选择和仪表盘指标足以衡量Entroly是否对你的工作负载有帮助。

### 🌐 联邦学习——实验性功能，可选加入

联邦是可选的。设计目标是共享匿名优化权重，而不是代码。

- 你的代码不应离开你的机器。共享的负载是优化统计/权重。
- 只有在你想参与跨安装学习实验时才启用。

```bash
export ENTROLY_FEDERATION=1
```

### ✂️ 响应蒸馏——输出也节省Token

LLM响应通常包含问候、含糊其辞和元评论。Entroly可以剥离常见填充内容，同时保留代码块不变。

```
之前: "当然！我很乐意帮忙。让我看看你的代码。
       问题出在auth模块。希望对你有帮助！"

之后: "问题出在auth模块。"
       → 减少输出token
```

三个强度等级：`lite` → `full` → `ultra`。一个环境变量即可启用。

### 🔒 本地索引；供应商请求在你的控制下

本地索引、选择、确定性验证和仪表盘不需要云服务。如果你代理云AI供应商，该供应商仍然会收到你通过Entroly发送的选定提示内容。在仅使用本地/离线命令和本地模型端点时，可用于物理隔离环境。

---

## 适配你的技术栈

| 工具 | 设置 |
|---|---|
| **Cursor** | `entroly init` → MCP server |
| **Claude Code** | `claude mcp add entroly -- entroly` |
| **VS Code (MCP)** | `entroly wrap vscode` |
| **Codex CLI** | `entroly wrap codex` |
| **Windsurf / Cline** | `entroly init` |
| **兼容的LLM API** | `entroly proxy` → HTTP代理 `localhost:9377` |

同样支持：OpenAI API • Anthropic API • LangChain • LlamaIndex • MCP-native

---

## 基准测试

### 示例进化追踪

来自本仓库本地开发vault的示例追踪：

```
[detect]     发现缺口 → entity="auth", miss_count=3
[synthesize] StructuralSynthesizer ($0, 确定性, 无LLM)
[benchmark]  skill=ddb2e2969bb0 → fitness 1.0 (1 pass / 0 fail, 338 ms)
[promote]    status: draft → promoted
[spend]      $0.0000 — 不变式 C_spent ≤ τ·S(t) 成立
```

### 准确率保持

在这些发布检查中，压缩后的上下文在统计上保持接近基线（gpt-4o-mini, Wilson 95% CI）：

| 基准测试 | n | 预算 | 基线 (95% CI) | 使用Entroly (95% CI) | 保持率 | Token节省 |
|---|---|---|---|---|---|---|
| NeedleInAHaystack | 20 | 2K | 100% [83.9–100%] | 100% [83.9–100%] | **100.0%** | **99.5%** |
| LongBench (HotpotQA) | 50 | 2K | 64.0% [50.1–75.9%] | 68.0% [54.2–79.2%] | **106.2%** | **85.3%** |
| Berkeley Function Calling | 50 | 500 | 100% [92.9–100%] | 100% [92.9–100%] | **100.0%** | **79.3%** |
| SQuAD 2.0 | 50 | 100 | 78.0% [64.8–87.2%] | 76.0% [62.6–85.7%] | **97.4%** | **39.3%** |
| GSM8K | 100 | 50K | 85.0% [76.7–90.7%] | 86.0% [77.9–91.5%] | **101.2%** | pass-through¹ |
| MMLU | 100 | 50K | 82.0% [73.3–88.3%] | 85.9% [77.8–91.4%] | **104.7%** | pass-through¹ |
| TruthfulQA (MC1) | 100 | 50K | 72.0% [62.5–79.9%] | 73.7% [64.3–81.4%] | **102.4%** | pass-through¹ |

> ¹ **pass-through**: 上下文已在预算范围内，Entroly保持不变。置信区间在这些运行中重叠；结果因模型、数据集、提示形状和token预算而异。

### Entroly与其他压缩方法对比（长上下文）

| 方法 | 保持率 | Token削减 | 架构 / 权衡 |
|---|---|---|---|
| **Entroly** | **100–106%** | **85–99%** | **极快（~80ms）。** 片段级背包算法保持完美的逐字结构保真度。 |
| Token级神经剪枝 | ~98–99% | 80–95% | **高开销。** 需要运行本地Transformer。Token级删除会破坏代码语法。 |
| 基于规则的逐字压缩 | ~100% | 50–70% | **高保真度。** 但Token削减较低。 |
| 注意力感知压缩 | 95%+ | 26–54% | **准确率稳健。** 但Token削减较低。 |

复现：`python -m bench.accuracy --benchmark all --model gpt-4o-mini --samples 100`

### CI/CD集成

在每个PR中运行token成本检查——在上线前捕获回退：

```yaml
- name: Check Entroly token budget
  run: pip install entroly && entroly batch --budget 8000 --fail-over-budget
```

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
  <b>使用本地、证据感知的工具测量和减少浪费的上下文token。</b><br/>
  <code>npm install -g entroly && entroly</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>

<p align="center">
  <a href="https://github.com/juyterman1000/entroly/discussions">讨论</a> •
  <a href="https://github.com/juyterman1000/entroly/issues">问题</a> •
  Apache 2.0 许可证
</p>
