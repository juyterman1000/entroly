<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/entroly_wordmark.svg" width="760" alt="Entroly">
</p>

<h1 align="center">Entroly — Context Assurance That Helps Lower AI Costs</h1>

<p align="center"><b>Use less unnecessary AI context without rebuilding how you work.</b><br>
Entroly selects useful evidence, removes avoidable repetition, keeps important originals recoverable, and records what your AI received.</p>

Entroly is a local-first Context Assurance layer for developers, teams, and AI applications. It works through supported proxy, wrapper, plugin, SDK, and MCP paths with Claude Code, Codex, OpenClaw, Hermes Agent, OpenCode, GitHub Copilot, Cursor, Aider, local models, and OpenAI/Anthropic-compatible applications.

A small one-time setup is required; no agent-architecture rewrite.

## Assurance-gated compression

The opt-in `compress_assured`, `compress_file_assured`, and
`compress_messages_assured` APIs return the
compressed output together with an accept/expand/bypass receipt. Structural
scope uses exact atomic source spans; semantic scope remains fail-closed without
a validated held-out calibration profile. The separate
`entroly-assurance-mcp` command exposes assured text/file compression, local decision
stats, repository impact, and whole-line context bundles.

See `docs/ASSURED_CONTEXT.md` in the source distribution for the full contract.

## What Entroly does

AI applications often send repeated files, long histories, logs, and low-value material to a model. Entroly can:

- select useful evidence under an explicit budget;
- remove duplicate or low-value context;
- preserve omitted originals through content-addressed recovery handles;
- create Context Receipts showing what was included, omitted, and risky;
- verify whether output claims are supported by supplied evidence;
- report provider-observed usage separately from local-only reductions;
- work across cloud and local models without locking you to one agent runtime.

Savings and quality depend on the workload, provider, model, budget, and integration. Entroly does not promise a universal compression percentage or guaranteed bill reduction.

## Install

```bash
pip install -U entroly
```

Check the local path before connecting a paid model:

```bash
entroly verify-claims
entroly simulate
entroly value
```

- `verify-claims` checks installation, receipts, recovery, and verification.
- `simulate` estimates context reduction locally without making a model call.
- `value` separates observed provider-bound usage from local-only reductions.

## Connect a supported AI tool

Automatic project setup:

```bash
cd /your/repo
entroly go
```

Examples:

```bash
# Claude Code
entroly attach create --client claude --project . --ttl 4h --install

# Codex
entroly attach create --client codex --project . --ttl 4h --install

# OpenClaw
entroly attach create --client openclaw --project . --ttl 4h --install

# Pay-as-you-go API or custom application
entroly proxy
```

For an MCP client, register the installed `entroly` command with no arguments:

```bash
entroly
```

Package-runner alternatives:

```bash
uvx --from entroly entroly
npx -y entroly-mcp
```

Claude Code MCP registration:

```bash
claude mcp add entroly -- uvx --from entroly entroly
```

Generic MCP configuration:

```json
{
  "mcpServers": {
    "entroly": {
      "command": "uvx",
      "args": ["--from", "entroly", "entroly"]
    }
  }
}
```

## How it works

```text
Files, documents, history, and tool output
                  ↓
       Entroly Context Assurance
       • rank useful evidence
       • select under budget
       • preserve exact originals
       • create a receipt
       • optionally verify output
                  ↓
        Your existing AI model
```

Local indexing, selection, receipts, and recovery storage do not require an Entroly cloud service. The selected prompt still goes to the provider configured by your application. No outbound analytics are enabled by default.

## Honest cost reporting

Entroly keeps different evidence classes separate:

| Path | What can be reported |
|---|---|
| Provider-bound proxy request | Observed pre/post input tokens and modeled input-cost avoidance with pricing provenance |
| SDK, MCP, plugin, or npm operation | Local context reduction; `$0` claimed when provider delivery is not observable |
| Fixed-price subscription | Context efficiency where supported; the subscription price may not change |
| Unknown or legacy history | Preserved operational history; excluded from savings claims |

Modeled provider cost avoidance is not a provider invoice.

## Technical capabilities

- budgeted evidence selection;
- recoverable compression;
- Context Receipts and Context Commits;
- strict `ccr:<24-hex>` exact-recovery handles;
- hash-only full-content retrieval with no semantic-query fallback;
- WITNESS evidence-support verification;
- pure-Python runtime with optional Rust acceleration;
- separate Node/WASM runtime;
- MCP, proxy, CLI, SDK, CI, OpenClaw, Hermes Agent, and OpenCode integrations;
- local-first operation and fail-open adapter behavior.

WITNESS evaluates support against supplied evidence; it does not establish universal truth.

## Everyday users

A future desktop experience, [Entroly Simple Mode](https://github.com/juyterman1000/entroly/blob/main/docs/product/entroly-simple-mode.md), is specified for people who do not want to learn about tokens, proxies, MCP, JSON, or agent architecture. It is not shipped yet, and Entroly does not claim a one-click consumer experience today.

## Links

- AI cost optimization: https://juyterman1000.github.io/entroly/docs/ai-cost-optimization.html
- Repository: https://github.com/juyterman1000/entroly
- Documentation: https://juyterman1000.github.io/entroly/docs/index.html
- Agent integrations: https://juyterman1000.github.io/entroly/docs/agent-integrations.html
- PyPI: https://pypi.org/project/entroly/
- npm runtime: https://www.npmjs.com/package/entroly
- npm MCP bridge: https://www.npmjs.com/package/entroly-mcp
- Evidence policy: https://github.com/juyterman1000/entroly/blob/main/docs/public-evidence.md
- Limitations: https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md

`mcp-name: io.github.juyterman1000/entroly`

Apache-2.0 licensed.
