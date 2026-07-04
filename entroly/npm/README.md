# entroly-mcp

An NPX MCP bridge for Entroly, the local context OS for AI coding agents.
Entroly gives AI workflows selected context, exact recovery handles, Context
Receipts, local verification, and token optimization.

## Which package should I use?

| You want | Use |
|---|---|
| Full Python CLI/SDK/MCP/proxy path | `pip install -U entroly` |
| Claude Code subscription setup | `claude mcp add entroly -- entroly` |
| MCP from npm with no Python dependency | `npx -y entroly-wasm serve` |
| MCP from npm, delegating to installed Python Entroly | `npx -y entroly-mcp serve` |
| Global Node/WASM CLI | `npm install -g entroly` |

The simplest first run for most users is still:

```bash
pip install -U entroly
entroly verify-claims
entroly simulate
```

Those commands run locally and do not call an LLM.

## MCP usage

Use one of these MCP configurations.

### Option A: Node/WASM only, no Python

```json
{
  "mcpServers": {
    "entroly": {
      "command": "npx",
      "args": ["-y", "entroly-wasm", "serve"],
      "env": {
        "ENTROLY_BUDGET": "200000"
      }
    }
  }
}
```

### Option B: NPX bridge to Python Entroly

Use this when you already installed the Python package and want the npm bridge
to delegate to it.

```json
{
  "mcpServers": {
    "entroly": {
      "command": "npx",
      "args": [
        "-y",
        "entroly-mcp",
        "serve"
      ],
      "env": {
        "ENTROLY_BUDGET": "200000"
      }
    }
  }
}
```

```bash
pip install -U entroly
# or:
pipx install entroly
```

### Claude Code subscription users

For Claude Code, the cleanest path is usually:

```bash
pip install -U entroly
claude mcp add entroly -- entroly
```

Claude Code stays your client. Entroly adds local MCP tools; you do not need to
run the proxy unless you control a provider API key and explicitly want proxy
mode.

### Features

- **Context Receipts**: Machine-readable JSON plus Markdown audit reports for selected and omitted context.
- **MCP Control Plane**: Local MCP tools for context selection, receipt rendering, omission explanations, checkpoints, and feedback.
- **Knapsack Token Optimization**: Fits the absolute maximum value into your token budget.
- **Shannon Entropy Scoring**: Prioritizes complex, high-entropy logic over repetitive boilerplate.
- **SimHash Deduplication**: Never wastes tokens on duplicate file contents.
- **Predictive Pre-fetch**: Learns your co-access patterns to predict what file you'll need next.
- **Feedback Loop**: Agentic feedback (`record_success` / `record_failure`) continuously tunes the RL weights.

## Context Receipt CLI

The NPX package forwards commands to the installed Python engine:

```bash
npx -y entroly-mcp ingest ./docs
npx -y entroly-mcp select --query "Does this contract have a change-of-control clause?" --budget 8000
npx -y entroly-mcp receipt .entroly/receipts/latest.json
npx -y entroly-mcp explain --why-omitted chk_abc123 --receipt .entroly/receipts/latest.json
```

These commands run locally and do not call an LLM.

## Links

- **PyPI**: [entroly](https://pypi.org/project/entroly/)
- **Repository**: [juyterman1000/entroly](https://github.com/juyterman1000/entroly)
