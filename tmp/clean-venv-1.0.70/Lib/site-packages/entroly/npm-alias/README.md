# entroly

Node/WASM package for Entroly, an auditable context engineering control plane
for AI agents with local compression, recovery, receipts, and verification.

This package installs the current Node/WebAssembly Entroly runtime by depending
on [`entroly-wasm`](https://www.npmjs.com/package/entroly-wasm), then exposes a
short `entroly` binary that delegates to `entroly-wasm`.

## Install

```bash
npm install -g entroly
```

Equivalent direct package:

```bash
npm install -g entroly-wasm
```

For the fullest CLI/SDK/proxy path, use the Python package:

```bash
pip install -U entroly
entroly verify-claims
entroly simulate
```

## Usage

```bash
entroly demo
entroly status
entroly serve
entroly optimize 8000 "fix the auth bug"
entroly health
entroly value
```

`entroly value` emits the same evidence-classified receipt shape as the Python
runtime. npm/MCP/local reductions report tokens with `$0` claimed unless a
provider-bound path is explicitly observed.

## MCP setup

For MCP clients that accept JSON config:

```json
{
  "mcpServers": {
    "entroly": {
      "command": "npx",
      "args": ["-y", "entroly-wasm", "serve"]
    }
  }
}
```

Claude Code subscription users usually get the smoothest path from the Python
package:

```bash
pip install -U entroly
claude mcp add entroly -- entroly
```

Proxy mode is optional and intended for users who control provider API keys.
