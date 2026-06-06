# entroly-mcp

An NPX bridge for Entroly, an auditable context control plane for AI agents.
Entroly gives AI workflows a Context Receipt: what was used, what was omitted,
why, and what risks remain.

## Installation & Usage

This package is a universal `npx` bridge to the Entroly Python engine.

You can use it directly in any MCP-compatible client like Cursor or Claude Desktop:

### Method 1: entroly-wasm (Recommended — zero dependencies)
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

### Method 2: NPX Bridge to Python (requires `pip install entroly`)
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

*Note: You must have the core Python engine installed on your system:*
```bash
pip install entroly
# or
pipx install entroly
```

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
