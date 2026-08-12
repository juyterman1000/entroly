# Privacy-Safe Product Health Telemetry

Entroly product-health telemetry is **off by default**. It starts only after an
operator gives explicit consent:

```bash
entroly telemetry preview
entroly telemetry on --endpoint https://telemetry.example/v1/events
entroly telemetry status
```

Without an endpoint, consent creates only a bounded local queue and uploads
nothing. `ENTROLY_DISABLE_TELEMETRY=1` and `ENTROLY_AIR_GAP=1` always override
stored consent. CI and test processes are excluded unless separately enabled.

## What is collected

The wire format is a closed allowlist. A consenting installation can report:

- Entroly version, operating-system family, and Python major/minor version
- a monthly rotating random pseudonym and UTC calendar day
- coarse surface starts: CLI, SDK, MCP, repository MCP, compression MCP, or proxy
- allowlisted CLI command, success/error/interrupted result, and duration bucket
- broad exception class such as `ValueError` or `OSError`, if error events remain enabled
- coarse token-reduction and reduction-percentage buckets
- whether the reduction was a local estimate or a provider-bound estimate
- whether a provider-bound reduction had a configured price and therefore a
  positive **modeled** cost signal

Command, surface, error, and value categories are deduplicated per UTC day.
Exact workload volume is not reported and event counts must not be interpreted
as command frequency.

## What is never collected

Entroly does not put any of the following into product-health events:

- prompts, source code, repository contents, or model inputs/outputs
- filenames, paths, repository names, project names, hostnames, or usernames
- exception messages, tracebacks, logs, request/response bodies, or environment values
- API keys, credentials, provider tokens, or telemetry tokens
- model identifiers, exact token counts, exact dollar amounts, or negotiated prices
- IP addresses, HTTP headers, or user-agent strings in the collector database

The network endpoint or reverse proxy necessarily sees a connection source IP
while accepting a request. The bundled collector does not read or store it and
runs with access logging disabled. Operators must also disable upstream access
logs or apply an appropriate short retention policy.

## Evidence boundaries

A positive token-reduction bucket means Entroly measured fewer estimated tokens
after its transformation. SDK and MCP-style local operations remain
`local_estimate`: Entroly cannot prove that their output reached a paid model.
Proxy transformations can be `provider_bound_estimate`; a positive modeled cost
signal additionally requires a configured rate for the provider-bound model.

Neither signal is a provider invoice, a billed-usage comparison, or a promise
that response quality improved. Reports keep these claims separate:

- registry downloads are package fetches, not people
- activation pseudonyms are explicitly consenting installations, not a census
- benefited pseudonyms observed a positive coarse token-reduction bucket
- modeled cost reduction is not verified money saved

## Storage, transport, and deletion

- Local queue: at most 200 events and 14 days, with owner-only permissions when supported
- Collector database: 90 days by default, configurable from 1 to 365 days
- Transport: HTTPS only, except explicit loopback HTTP for local development
- Ambient proxy variables: ignored unless `ENTROLY_TELEMETRY_TRUST_PROXY_ENV=1`
- Upload batch: at most 20 events and 64 KiB with a short fail-open timeout;
  automatic attempts occur at most once per UTC day
- Withdrawal: `entroly telemetry off` removes local consent, queue, status,
  markers, and random seed, and requests deletion of the four recent monthly
  pseudonyms from a configured collector

## Self-host the aggregate collector

The collector binds to loopback and should sit behind an authenticated HTTPS
reverse proxy. In PowerShell:

```powershell
$env:ENTROLY_TELEMETRY_INGEST_TOKEN = "replace-me"
$env:ENTROLY_TELEMETRY_ADMIN_TOKEN = "replace-me-too"
python -m entroly.telemetry_collector --db C:\entroly-data\product-health.db
```

On macOS or Linux, use `export NAME=value`. Keep the admin summary endpoint
private. If ingest is intentionally tokenless for public clients, enforce
request-size limits and rate limiting at the reverse proxy.

The collector exposes only aggregate summaries: active monthly pseudonym counts,
activation counts, coarse errors, command reliability, platform-family health,
and coarse benefit evidence. It never returns raw events or pseudonyms.

Telemetry cannot observe a package installation that fails before Entroly first
starts. Registry downloads, release CI, and user-submitted diagnostics must cover
that blind spot; the report must not misclassify a download as a working install.

To combine those aggregate observations with registry downloads without
pretending downloads are users:

```bash
python scripts/adoption_report.py \
  --collector-summary-url https://telemetry.example/v1/summary?days=30
```

The report labels the resulting ratios as observed opt-in diagnostics, never as
an actual unique-user adoption rate.

## Structured exit feedback

Python and modern npm package managers do not provide a dependable Entroly
uninstall callback. Pip documents package removal as a package-manager action,
and npm explicitly states that uninstall lifecycle scripts are not implemented:

- [pip uninstall reference](https://pip.pypa.io/en/stable/cli/pip_uninstall/)
- [npm lifecycle scripts](https://docs.npmjs.com/cli/using-npm/scripts/#a-note-on-a-lack-of-npm-uninstall-scripts)

Entroly therefore provides an honest guided path instead of pretending every
removal can be observed:

```bash
# Python installation: optional survey, then pip uninstall
entroly uninstall

# Inspect the exact response and command without sending or changing anything
entroly uninstall --dry-run \
  --reason runtime_error --benefit no --surface mcp --duration 1_7d

# npm, Docker, or another removal path: survey only, then use that manager
entroly uninstall --feedback-only
npm uninstall -g entroly
```

The survey has no free-text field. It contains only:

- one reason from a fixed list
- self-reported token-reduction benefit: yes, no, unsure, or not measured
- one primary product surface
- one coarse use-duration bucket
- the same release, OS-family, Python major/minor, UTC-day, and pseudonym rules
  described above

Interactive sending defaults to **No** and shows the collector origin first.
For non-interactive use, all four fields and `--send-feedback` are required.
`ENTROLY_FEEDBACK_ENDPOINT` or `--endpoint` selects the HTTPS collector, but
neither causes a submission without the interactive confirmation or explicit
flag. Air-gap and hard-disable policies still win.

The response is sent synchronously once and is never queued on failure. When
ongoing telemetry is enabled and the destination is the same configured
collector, it uses the current monthly pseudonym so the private aggregate report
can count how many exiting installations previously observed a benefit or error.
A different or one-time collector receives a random one-event pseudonym that is
never persisted. The guided flow revokes local telemetry consent before the
package is removed, so a later reinstall requires fresh consent.

`--delete-remote-telemetry` requests deletion of recent linked telemetry before
sending any separately confirmed one-time response. Collector retention still
applies to that new response. Users who do not want a retained exit response
should decline the survey; no uninstall behavior is affected.

The private aggregate summary exposes exit-response counts, fixed reasons,
self-reported benefit, primary surfaces, duration buckets, OS-family counts,
and counts with prior benefit/error observations. It never exposes response
rows, pseudonyms, or cross-tabs containing machine-identifying data. Direct
`pip uninstall`, `npm uninstall`, deleted containers, and failed installations
remain invisible; registry downloads must never be used to fabricate an exit or
retention rate.
