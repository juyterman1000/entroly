# Entroly Quickstart

This path proves the installed package locally before connecting an API key or
changing an agent configuration. Expected time: three to five minutes.

## 1. Install

```bash
python -m pip install --upgrade entroly
```

Python 3.10 or newer is required. For Node-first environments, use
`npm install -g entroly`; the Python distribution currently exposes the broadest
CLI and SDK surface.

## 2. Prove the installation

Run these commands inside a repository you can safely inspect:

```bash
entroly --version
entroly doctor
entroly verify-claims
entroly simulate
```

`verify-claims` exercises the packaged local path and writes a machine-readable
report. `simulate` estimates token reduction on the current repository without
calling a model. If the repository is already small or under budget, passthrough
or low savings is a valid result.

## 3. Choose one integration

Do not configure every mode at once. Start with the path that matches how the
model is invoked.

### Existing Claude Code, Codex, or OpenClaw session

```bash
entroly attach create --client claude --project . --ttl 4h --install
```

Replace `claude` with `codex` or `openclaw`. The grant is project-scoped,
expiring, and revocable:

```bash
entroly attach list
entroly attach revoke <grant-id> --uninstall
```

### MCP client

Run `entroly init`, or register `entroly` as a local stdio MCP command with no
arguments. Restart the client, list its MCP tools, and run a small optimization
before enabling it on a large repository.

### Application using provider API keys

Install proxy dependencies and start the local proxy:

```bash
python -m pip install --upgrade "entroly[proxy]"
entroly proxy
```

Point only the application being tested at `http://localhost:9377` using its
provider-specific base-URL setting. Do not overwrite a global base URL until the
smoke test succeeds.

### Python SDK

```python
from entroly import compress

source = "INFO ready\n" * 100 + "ERROR connection refused\n"
smaller = compress(source, budget=80, content_type="log")
print(smaller)
```

For a real task query, prefer `optimize(fragments, budget=..., query=...)` over
query-agnostic compression.

## 4. Confirm the result

```bash
entroly status
entroly dashboard
```

For receipt-producing paths, inspect selected and omitted evidence, warnings,
token ratios, and recovery status before relying on the compressed context.

## Stop and troubleshoot when

- the original request is not preserved after an optimization error;
- `doctor` or `verify-claims` fails;
- a client cannot list the expected MCP tools;
- a receipt is missing or cannot explain an omission;
- provider streaming, tools, headers, or model parameters change unexpectedly.

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and file a redacted
[bug report](https://github.com/juyterman1000/entroly/issues/new?template=bug_report.yml)
if recovery is not straightforward.

## Next steps

- [Examples](EXAMPLES.md)
- [Best practices](BEST_PRACTICES.md)
- [API reference](API_REFERENCE.md)
- [Architecture](ARCHITECTURE.md)
- [Limitations](docs/limitations.md)
