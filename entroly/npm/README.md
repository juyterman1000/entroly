# @ebbiforge/entroly-mcp

An MCP (Model Context Protocol) server for information-theoretic context optimization. 
Use it to optimize Claude/Cursor's context window by selecting only the most relevant, high-entropy, and critical code fragments based on a Knapsack constraint.

## Installation & Usage

This package is a universal `npx` bridge to the Entroly Python engine.

You can use it directly in any MCP-compatible client like Cursor or Claude Desktop:

### Method 1: NPX (Claude Desktop/Cursor)
```json
{
  "mcpServers": {
    "entroly": {
      "command": "npx",
      "args": [
        "-y",
        "@ebbiforge/entroly-mcp"
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

- **Knapsack Token Optimization**: Fits the absolute maximum value into your token budget.
- **Shannon Entropy Scoring**: Prioritizes complex, high-entropy logic over repetitive boilerplate.
- **SimHash Deduplication**: Never wastes tokens on duplicate file contents.
- **Predictive Pre-fetch**: Learns your co-access patterns to predict what file you'll need next.
- **Feedback Loop**: Agentic feedback (`record_success` / `record_failure`) continuously tunes the RL weights.

## Links

- **PyPI**: [entroly](https://pypi.org/project/entroly/)
- **Repository**: [juyterman1000/entroly](https://github.com/juyterman1000/entroly)
