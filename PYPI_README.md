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

For an MCP client, register the installed `entroly` command with no arguments.
When an MCP client launches it with a stdio pipe, Entroly starts the installed
Python server directly:

```bash
entroly
```

Or register a package runner, also with no `serve` argument:

```bash
uvx --from entroly entroly
npx -y entroly-mcp
```

`entroly serve` is a different deployment path: it uses the Entroly Docker
image by default. For the installed Python runtime in an interactive shell, use
`ENTROLY_NO_DOCKER=1 entroly serve` on macOS/Linux or set
`ENTROLY_NO_DOCKER=1` in the client environment.

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
      "args": ["--from", "entroly", "entroly"]
    }
  }
}
```

External MCP galleries can lag a release. Direct stdio registration above is
the canonical setup; confirm a gallery entry's package version and validation
status before relying on it.

### Claude Code

```bash
claude mcp add entroly -- uvx --from entroly entroly
```

### Generic MCP configuration

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

## What Entroly adds

- **Context selection** under explicit token and cost budgets
- **Context Commits** linking selected, omitted, and recoverable evidence
- **Context Receipts** for replay, audit, and omission explanations
- **Exact recovery** of compressed fragments through stable handles
- **Local verification** through WITNESS and receipt checks
- **Context Check** coverage evidence for changed files and CI risk gates
- **Pure-Python base runtime**, optional Rust acceleration, and a separate npm/WASM runtime
- **Local-first operation** with no outbound analytics by default

## Links

- Repository: https://github.com/juyterman1000/entroly
- Documentation: https://juyterman1000.github.io/entroly/
- PyPI: https://pypi.org/project/entroly/
- npm runtime: https://www.npmjs.com/package/entroly
- npm MCP bridge: https://www.npmjs.com/package/entroly-mcp
- Public evidence policy: https://github.com/juyterman1000/entroly/blob/main/docs/public-evidence.md

## MCP Registry identity

`mcp-name: io.github.juyterman1000/entroly`

Apache-2.0 licensed.
