# Entroly submission kit

Use the narrowest copy that fits the target. Preserve links and factual limits.
Do not add benchmark or adoption numbers unless the target requires them and the
submission links the exact current artifact.

## Canonical identity

- **Name:** Entroly
- **Repository:** https://github.com/juyterman1000/entroly
- **Documentation:** https://juyterman1000.github.io/entroly/docs/index.html
- **License:** Apache-2.0
- **Primary install:** `pip install -U entroly`
- **Primary npm install:** `npm install -g entroly`
- **MCP manifest:** `server.json`
- **Claude plugin manifest:** `.claude-plugin/manifest.json`
- **AI-readable product summary:** `llms.txt`

## Default one-line description

> Entroly is an open-source, local-first Context Assurance layer for AI agents
> that selects evidence under a token budget, keeps omitted source recoverable,
> emits auditable Context Receipts, and verifies model claims against supplied
> evidence.

## Short descriptions

### Claude Code and coding-agent catalogs

> Entroly adds local context selection, recoverable compression, Context
> Receipts, and evidence verification to Claude Code and other coding agents
> through MCP, proxy, wrapper, plugin, and SDK paths.

### MCP catalogs

> Entroly is a local-first MCP Context Assurance server for budget-aware evidence
> selection, recoverable compression, repository context, Context Receipts, and
> evidence-grounding checks.

### CLI catalogs

> Entroly is a local-first CLI for inspecting, selecting, compressing,
> receipting, recovering, and verifying context before it reaches an AI agent.

### Local AI catalogs

> Entroly runs indexing, selection, compression, receipt generation, recovery,
> and default evidence verification locally. Proxy mode forwards only the
> selected request to the user's configured model provider.

### LLMOps and agent-infrastructure landscapes

> Entroly is an open-source context-control plane for agents. It combines
> budget-aware evidence selection, recoverable compression, Context Receipts,
> verification, proxy controls, MCP, and SDK integration without replacing the
> agent or model runtime.

## Suggested entries

### Claude Code Ultimate Guide

```markdown
- [Entroly](https://github.com/juyterman1000/entroly) — Local-first Context
  Assurance for Claude Code via MCP, proxy, wrapper, and plugin paths. Selects
  evidence under a token budget, keeps omitted source exactly recoverable, emits
  Context Receipts, and verifies claims against supplied evidence.
```

### awesome-claude-code

```markdown
- [Entroly](https://github.com/juyterman1000/entroly) — Context Assurance and
  recoverable context compression for Claude Code, with MCP attachment, scoped
  recovery, receipts, and local evidence verification.
```

### awesome-claude-plugins

```markdown
- [Entroly](https://github.com/juyterman1000/entroly) — Claude plugin and MCP
  integration for budget-aware repository context, recoverable compression,
  Context Receipts, and evidence verification. Apache-2.0.
```

Verification surfaces:

- `.claude-plugin/manifest.json`
- `server.json`
- `README.md`

### awesome-claude MCP page

```mdx
---
title: Entroly
description: Local-first Context Assurance for AI agents with budget-aware evidence selection, recoverable compression, Context Receipts, and verification.
repository: https://github.com/juyterman1000/entroly
license: Apache-2.0
---

Entroly exposes MCP tools for selecting and compressing context, recording
receipts, recovering omitted source through content-addressed handles, and
checking model claims against supplied evidence. Indexing and default
verification run locally. Install with `pip install -U entroly` or use the
`entroly-mcp` npm bridge.
```

### Awesome-MCP-ZH

```markdown
- [Entroly](https://github.com/juyterman1000/entroly) - 面向 AI Agent 的本地优先
  Context Assurance 与 MCP 服务。支持基于 token 预算的证据选择、可恢复压缩、
  Context Receipts、内容寻址恢复和证据验证。索引、选择、压缩及默认验证在本地运行；
  代理模式仅将筛选后的请求发送到用户配置的模型服务商。
```

Do not translate benchmark percentages into the listing unless the external
maintainer explicitly requests linked measurements.

### awesome-cli-apps

```markdown
- [Entroly](https://github.com/juyterman1000/entroly) — Inspect, select,
  compress, recover, and verify AI-agent context locally, with CLI, MCP, proxy,
  and SDK interfaces.
```

### awesome-local-llms

```markdown
- [Entroly](https://github.com/juyterman1000/entroly) — Local-first context
  control for AI agents: budget-aware evidence selection, recoverable
  compression, receipts, and local evidence verification. Can be used with
  local models or as a proxy in front of configured providers.
```

### Agentic AI landscape

```text
Entroly | Context engineering / agent infrastructure | Open source | Apache-2.0
Local-first context-control plane with budget-aware evidence selection,
recoverable compression, Context Receipts, MCP, proxy, SDK, and evidence
verification.
```

## Reviewer kit

For an independent hands-on review, ask the reviewer to record:

1. Entroly version and installation source.
2. Exact input workload and repository revision.
3. Raw-context baseline and Entroly treatment.
4. Model, provider, token budget, cache behavior, and pricing assumptions.
5. Input and output token counts from the provider or harness.
6. Task outcome, not just compression ratio.
7. Recovery and receipt behavior.
8. Failures, regressions, and workloads where Entroly passes through.
9. Raw artifacts and commands needed to reproduce the result.

Recommended no-key first run:

```bash
pip install -U entroly
cd /path/to/repository
entroly verify-claims
entroly simulate
```

A paid model comparison should not start until the task, baseline, token cap,
scoring rule, and null/control arms are frozen.

## Neutral benchmark contribution contract

An Entroly benchmark adapter must:

- pin the exact Entroly version;
- use the upstream workload and scoring code unchanged;
- preserve identical model, provider, temperature, prompt, and token cap across
  comparable arms;
- record whether Entroly used selection, compression, recovery, verification, or
  pass-through;
- retain raw outputs and failure classifications;
- include a no-Entroly baseline and any required null control;
- avoid tuning on the held-out result set;
- report quality and cost/token effects together.

A larger reduction with lower task quality is not a win. A ceiling where all
arms solve every task is not evidence of non-inferiority.

## Package-manager submission contract

Before requesting inclusion in Nix, Scoop, FreeBSD Ports, or another maintained
package collection, verify:

- a stable released source or binary artifact exists for that platform;
- checksums are derived from the released artifact;
- installation is non-interactive and uninstall behavior is documented;
- the package does not make surprise network calls;
- update automation or a clear maintainer path exists;
- license and dependency metadata are complete;
- a clean-machine smoke test passes.

## Claims not authorized for generic submissions

Do not write any of the following without a directly linked, current,
workload-specific artifact and caveat:

- "up to 90% fewer tokens";
- "zero quality loss";
- "better than Headroom";
- "better than LeanCTX";
- "production proven";
- "works with every agent";
- download, star, user, or installation counts;
- universal cost savings.

Prefer product mechanics that a maintainer can inspect in the repository over
marketing adjectives.
