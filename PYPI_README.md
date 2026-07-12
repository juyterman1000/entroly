<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/entroly_wordmark.svg" width="760" alt="Entroly">
</p>

<p align="center"><b>Know exactly what your AI agent saw.</b></p>

Entroly is a local context control plane for AI coding agents. It selects the
highest-value evidence under a token budget, records what was selected and
omitted, keeps compressed context recoverable, and produces verifiable Context
Commits and receipts.

## Install

```bash
pip install -U entroly
```

Run the local, no-key verification path:

```bash
entroly verify-claims
entroly simulate
```

## MCP server

Start the stdio MCP server directly:

```bash
entroly serve
```

Or use a package runner:

```bash
uvx --from entroly entroly serve
npx -y entroly-mcp serve
```

Entroly works with GitHub Copilot in VS Code, Claude Code, Cursor, Windsurf,
Cline, Continue, Zed, and other MCP-compatible clients.

### GitHub Copilot / VS Code

Create `.vscode/mcp.json`:

```json
{
  "servers": {
    "entroly": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "entroly", "entroly", "serve"]
    }
  }
}
```

Or install through the MCP gallery after the official registry listing becomes
available by searching for **Entroly**.

### Claude Code

```bash
claude mcp add entroly -- uvx --from entroly entroly serve
```

### Generic MCP configuration

```json
{
  "mcpServers": {
    "entroly": {
      "command": "uvx",
      "args": ["--from", "entroly", "entroly", "serve"]
    }
  }
}
```

## What Entroly adds

- **Context selection** under explicit token and cost budgets
- **Context Commits** linking selected, omitted, and recoverable evidence
- **Context Receipts** for replay, audit, and omission explanations
- **Exact recovery** of compressed fragments through stable handles
- **Local verification** through WITNESS and receipt checks
- **Context Check** coverage evidence for changed files and CI risk gates
- **Rust and WASM engines** with a pure-Python fallback
- **Local-first operation** with no outbound analytics by default

## Links

- Repository: https://github.com/juyterman1000/entroly
- Documentation: https://juyterman1000.github.io/entroly/
- PyPI: https://pypi.org/project/entroly/
- npm MCP bridge: https://www.npmjs.com/package/entroly-mcp

## MCP Registry identity

`mcp-name: io.github.juyterman1000/entroly`

Apache-2.0 licensed.
