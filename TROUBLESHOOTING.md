# Troubleshooting

Start with evidence. Do not delete caches, indexes, receipts, or configuration
until you have captured the failure and confirmed what recovery data is needed.

## Baseline diagnostics

```bash
entroly --version
entroly doctor
entroly verify-claims
entroly status
```

Record the install method, operating system, architecture, Python or Node
version, agent/client version, exact command, expected result, and actual result.
Redact provider keys, bearer tokens, private paths, source text, receipts, and
recovery bundles before sharing output.

## Command is not found

1. Confirm the environment used for installation is active.
2. Run `python -m pip show entroly` and `python -m pip --version` from the same
   shell.
3. Reopen the shell after a global installer changes `PATH`.
4. Prefer `python -m pip install --upgrade entroly` to avoid a mismatched `pip`.
5. For an MCP client that does not inherit `PATH`, configure the absolute
   executable path returned by `Get-Command entroly` on Windows or
   `command -v entroly` on macOS/Linux.

## Native engine is unavailable

The base package can operate without native acceleration.

```bash
python -m pip install --upgrade "entroly[native]"
entroly doctor
entroly verify-claims
```

If no wheel exists for the environment, use the pure-Python path or build from
source with a Rust toolchain and maturin. Do not copy a native module between
Python versions. Report the capability status from `verify-claims`, not only an
import result.

## MCP tools do not appear

1. Run `entroly doctor` outside the client.
2. Verify the client launches `entroly` as a stdio command with no shell-only
   aliases.
3. Restart or reload the client after editing MCP configuration.
4. Inspect the client's MCP logs for executable, JSON, and permission errors.
5. For scoped attachment, run `entroly attach list` and verify the grant is not
   expired, revoked, for another project, or missing the requested scope.

Do not place a bearer token directly in a process argument if the attachment
flow supports a scoped local credential.

## Proxy starts but requests fail

```bash
python -m pip install --upgrade "entroly[proxy]"
entroly proxy
entroly status
```

- Change the base URL only for the application being tested.
- Preserve the provider API key in the provider's normal environment variable.
- Confirm the correct protocol suffix, such as `/v1` where the client requires
  it.
- Test a non-streaming request, then streaming and tool calls.
- If an optimized request is rejected, capture whether exact passthrough was
  attempted and whether the original provider status was preserved.

## No savings or more tokens than expected

No savings can be correct for small, unique, or already compact input.

```bash
entroly simulate
entroly perf
```

Check the original token estimate, selected integration, explicit budget,
content type, receipt warnings, and whether provider-observed usage agrees with
the local estimate. Do not compare estimated input reduction with billed total
cost or output-token reduction.

## Answer quality dropped

1. Stop using aggressive compression on the affected workload.
2. Preserve the original input, query, selected context, receipt, output, model
   settings, and exact versions.
3. Retry with exact passthrough and a more conservative budget.
4. Inspect omitted relevant chunks and dependency links.
5. File an evidence report with per-case artifacts that contain no private data.

One failing held-out case is more useful than an aggregate complaint.

## Omitted evidence cannot be recovered

Exact recovery requires a receipt created on a recoverable path and its matching
local recovery store. Verify the receipt fingerprint and store location. If the
store was deleted or the fingerprint fails, Entroly must not claim exact
recovery. Restore from a trusted backup or rerun selection against the original
source; do not treat re-derived text as byte-exact recovery.

## Dashboard shows zero, unavailable, or stale values

Zero is different from unavailable. Confirm the producing process is running,
the dashboard is reading the same Entroly state directory, and the ledger has a
recent event. Forecast, local estimate, provider-observed usage, and invoice
savings are separate tiers and should not be combined.

## OpenClaw plugin problems

```bash
openclaw plugins list
openclaw plugins install clawhub:entroly-openclaw
```

Confirm the installed plugin version, OpenClaw host version, Node.js 22.19 or
newer, and configured context-engine selection. OpenClaw remains responsible for
provider routing and authentication. Include the plugin status receipt and a
redacted normalized message shape when reporting a bridge failure.

## Upgrade or release mismatch

Registry and marketplace pages can update at different times. Verify PyPI, npm,
GitHub Release, Docker/GHCR, ClawHub, and Homebrew independently. A Git tag or a
successful local build does not prove publication. Do not update a Homebrew
checksum until the matching PyPI sdist is live.

## Safe bug-report template

```text
Entroly version and install method:
OS / architecture / Python or Node version:
Agent, client, or provider path:
Exact command:
Expected behavior:
Actual behavior:
Was original context preserved?:
Was any data lost or unrecoverable?:
Receipt or diagnostic warning (redacted):
Minimal synthetic reproduction:
```

Use the [Entroly bug form](https://github.com/juyterman1000/entroly/issues/new?template=bug_report.yml)
or [SUPPORT.md](SUPPORT.md) for the appropriate channel.
