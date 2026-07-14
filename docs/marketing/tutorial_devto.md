# Auditable context control for AI coding agents with Entroly

> Maintainer draft. Re-run every command and review the linked evidence before
> publishing. Do not add a universal savings or accuracy claim.

Long coding sessions accumulate logs, duplicate files, and stale conversation
history. Sending everything can be expensive; deleting context blindly can
remove the evidence the model needs. Entroly provides a local middle path:
budgeted selection, recoverable omissions, and receipts describing what was
selected and left out.

## Install and verify locally

```bash
pip install -U entroly
entroly verify-claims
entroly simulate
```

`verify-claims` is a bounded package and recovery smoke test. `simulate` is a
local token estimate for the current repository. Neither command proves a bill
reduction or downstream answer quality.

## Connect an MCP client

For supported IDEs, let Entroly generate the configuration:

```bash
entroly init
```

For a generic MCP client, register the installed command with no arguments:

```json
{
  "mcpServers": {
    "entroly": {
      "command": "entroly",
      "args": []
    }
  }
}
```

Under an MCP client's stdio pipe, the bare command starts the installed Python
server. `entroly serve` is Docker-first unless `ENTROLY_NO_DOCKER=1` is set.

Claude Code users can register the same command directly:

```bash
claude mcp add entroly -- entroly
```

## Use proxy mode only when it fits

If you control a provider API key and the client supports a custom endpoint:

```bash
entroly proxy
```

Then configure only the provider base URL supported by that client. Entroly
performs local selection, but the resulting prompt still goes to the model
provider configured by the user. Cache alignment can preserve eligible prefix
bytes; only provider-reported usage can establish that a cache hit occurred.

## Measure before making a claim

Record:

- the raw request and Entroly request under the same task;
- provider-observed input, cached, and output tokens;
- model and provider;
- context budget and Entroly version;
- task success and answer-critical evidence recall; and
- retries, recovery calls, and omitted evidence.

The repository includes a preregistered
[Context Efficiency Frontier](../benchmarks/context-efficiency-frontier.md)
protocol for paired evaluation. Public benchmark numbers and their caveats live
in the [public evidence policy](../public-evidence.md).

## What to say accurately

Entroly is an Apache-2.0, local context-control plane with Python, optional Rust
acceleration, MCP and proxy integrations, and a separate Node/WASM runtime. It
can reduce selected context on some workloads, but the amount and any effect on
answer quality depend on the corpus, query, budget, model, and integration.
