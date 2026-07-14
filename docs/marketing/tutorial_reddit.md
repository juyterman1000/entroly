# Community post templates

> Maintainer drafts. Follow each community's self-promotion rules, disclose your
> relationship to Entroly, and answer questions with links to exact evidence.
> Do not post a universal savings, cache-hit, latency, or accuracy claim.

## Technical project post

**Title:** Entroly: open-source context receipts and recoverable selection for AI agents

**Body:**

> I maintain Entroly, an Apache-2.0 local context-control plane for AI agents.
> It selects evidence under a budget, records selected and omitted context, and
> can keep omitted source recoverable through receipts and handles.
>
> The base Python install has a pure-Python path; Rust acceleration is optional,
> and the Node package uses a separate WASM runtime. MCP clients register the
> bare `entroly` command with no arguments.
>
> I am deliberately not posting a universal token-savings percentage. The repo
> links each prominent benchmark number to its exact result and caveats, and it
> includes a paired protocol for measuring provider-observed usage and task
> quality. I would value feedback on the receipt format, recovery model, and
> reproducibility rather than stars alone.

Links:

- Repository: `https://github.com/juyterman1000/entroly`
- Public evidence policy: `https://github.com/juyterman1000/entroly/blob/main/docs/public-evidence.md`
- Limitations: `https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md`

## Setup-focused response

```bash
pip install -U entroly
entroly verify-claims
entroly simulate
```

For an MCP client:

```json
{
  "mcpServers": {
    "entroly": { "command": "entroly", "args": [] }
  }
}
```

Explain that `verify-claims` is a bounded local smoke test and `simulate` is an
estimate. Ask the user to share the generated JSON and exact client/version if
setup fails. Do not tell them a passing smoke test proves production savings.

## Maintainer response rules

1. State that you maintain or contribute to Entroly.
2. Link the exact result file for any number.
3. Distinguish pure-Python, optional Rust, and Node/WASM paths.
4. Distinguish MCP from the HTTP proxy.
5. Say when a result is estimated rather than provider-observed.
6. Treat bug reports as trust failures and provide a reproducible recovery path.
