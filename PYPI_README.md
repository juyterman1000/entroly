<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/entroly_wordmark.svg" width="760" alt="Entroly">
</p>

<h1 align="center">Entroly — The Open-Source Context OS for AI Agents</h1>

<p align="center"><b>Keep your agent. Give it a Context OS.</b><br>
The observability, governance, and decision layer for AI context.</p>

Entroly is an open-source Context OS for AI agents: auditable context
engineering, recoverable compression, memory, verification, provider controls,
receipts, security, and guarded outcome learning in one local layer.

## Install

```bash
pip install -U entroly
```

Run the local, no-key verification path:

```bash
entroly verify-claims
entroly simulate
entroly value
```

`entroly value` keeps provider-bound cost avoidance separate from SDK, MCP,
and npm reductions. Local-only operations report tokens reduced with `$0`
claimed; modeled provider cost avoidance includes pricing provenance and is
not a provider invoice.

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

Entroly works with Claude Code, Codex, OpenClaw, GitHub Copilot in VS Code,
Cursor, Windsurf, Cline, Continue, Zed, and other MCP-compatible clients.

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
- **Proof-guided recovery** that verifies drafts, recovers exact omitted
  evidence, and stops under declared round/token bounds; local prepare and
  advance operations never call a provider
- **Local verification** through WITNESS and receipt checks
- **Context Check** coverage evidence for changed files and CI risk gates
- **Verified model-based dreaming** (experimental, opt-in): real transitions
  train the model, synthetic rollouts only rank experiments, and real holdout
  evidence remains mandatory for promotion
- **Pure-Python base runtime**, optional Rust acceleration, and a separate npm/WASM runtime
- **Local-first operation** with no outbound analytics by default

Prepare a restart-safe model request without a provider call:

```bash
entroly proof prepare ./docs --query "What evidence supports this answer?" \
  --budget 8000 --idempotency-key request-001
```

The caller sends the returned request through its existing model route and
returns the draft with `entroly proof advance`. See the repository's
proof-guided protocol guide for MCP, proxy, and opt-in OpenClaw automation.

## Links

- Repository: https://github.com/juyterman1000/entroly
- Documentation: https://juyterman1000.github.io/entroly/docs/index.html
- PyPI: https://pypi.org/project/entroly/
- npm runtime: https://www.npmjs.com/package/entroly
- npm MCP bridge: https://www.npmjs.com/package/entroly-mcp
- Public evidence policy: https://github.com/juyterman1000/entroly/blob/main/docs/public-evidence.md

## MCP Registry identity

`mcp-name: io.github.juyterman1000/entroly`

Apache-2.0 licensed.
