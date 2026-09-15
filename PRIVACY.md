# Privacy Policy

Local indexing, context selection, and deterministic verification run on your
machine. Cloud-provider requests and explicitly enabled integrations can transmit
data. Product telemetry, update checks, and runtime native-engine repair are off
by default. This policy describes this revision; older releases may differ.

## Network recipients and controls

| Feature | Recipient and data | Activation |
| --- | --- | --- |
| Cloud LLM proxy | Configured provider receives requests, including selected code, prompts, and tool data. Its data policy applies. | Operator configures and uses a provider route. |
| WITNESS NLI | OpenAI receives verification prompts, evidence, and extracted claims. | `--witness-nli` or `ENTROLY_WITNESS_NLI=1`, with credentials. |
| Native repair | Configured package index (normally PyPI) and package hosts receive installer requests. Entroly does not attach repository content. | Explicit `entroly.repair()` or startup opt-in `ENTROLY_ENABLE_SELF_HEAL=1`. |
| Update check | PyPI receives a version request, at most once per 24 hours. | `ENTROLY_ENABLE_UPDATE_CHECK=1`; `ENTROLY_DISABLE_UPDATE_CHECK=1` overrides. |
| Product telemetry | Configured collector receives allowlisted coarse events and a rotating pseudonym. | `entroly telemetry on` gives consent; upload requires an endpoint. |
| Federation | Configured file exchange or GitHub receives noised weights and a random persisted client ID. Personal-token GitHub transport exposes the posting account. | `ENTROLY_FEDERATION=1` and configured transport. |
| Notifications | Configured Slack/Discord webhooks or Telegram receive notification payloads. | Operator supplies webhook or bot configuration. |

Package managers can honor custom indexes, mirrors, credentials, and proxies;
review their configuration before allowing installation. `ENTROLY_NO_SELF_HEAL=1`
and `ENTROLY_AIR_GAP=1` override explicit and automatic repair. Imports never
trigger repair. Without a usable native engine, affected selection paths report
degraded behavior and label budget-only reduction figures as unearned.

Local preprocessing does not prevent the proxy sending code to your provider.
Optional RAVS model substitution (`ENTROLY_RAVS_ROUTER=1`) stays within the original
provider on the live gateway path. See [provider boundaries](docs/gateway-provider-boundary.md).
Active escalation can repeat a request to the configured provider. Startup
provider probing requires `ENTROLY_CHECK_UPSTREAM=1`.

Feature flags are not an operating-system network sandbox. Strict offline
operation also requires network isolation and review of custom tools, provider
routes, package managers, and configured integrations.

## Product telemetry

Telemetry is **off by default**. Consent without an endpoint creates a bounded
local queue only. With an endpoint, version/platform categories, coarse usage and
reduction buckets, and monthly rotating pseudonyms can be uploaded. The event
schema excludes code, prompts, paths, credentials, exception messages, and
request/response bodies. The receiving network infrastructure can still see the
source IP. See the complete [telemetry policy](docs/telemetry-privacy.md).

```bash
entroly telemetry preview
entroly telemetry status
entroly telemetry off
```

`ENTROLY_DISABLE_TELEMETRY=1` and `ENTROLY_AIR_GAP=1` override stored consent.
Withdrawal removes local telemetry state and requests deletion of recent
pseudonyms from the configured collector. It does not prove that an unreachable
collector completed deletion. Contributions already folded into identifier-free
cumulative counters cannot be linked back and individually subtracted.

## Local storage and recovery

State can include code, original prompts, paths, receipts, Context Commits,
vaults, logs, and recovery data. Locations depend on the surface, workspace,
explicit store path, and `ENTROLY_DIR`. Some defaults live under `~/.entroly/`;
checkpoint creation can fall back to an OS temporary directory. Deleting the
current project's `.entroly/` is not complete erasure.

Recovery data is **not encrypted by Entroly**. Exact recovery retains input;
a secret detector cannot guarantee that stored originals contain no secrets.
Keep artifacts as private as their source and inspect them before sharing.
See [recovery-data security](docs/recovery-data-security.md) for locations,
permissions, atomic writes, scope limitations, retention, and deletion.
Receipt and Context Commit retention is operator-managed. Telemetry expiry
does not automatically remove source recovery material.

## Tools and verification limits

Promoted `tool.py` skills execute only with `ENTROLY_EXECUTE_PROMOTED_SKILLS=1`.
They run in subprocesses with a timeout, which is not a security sandbox. Review
executable vault state and bound its filesystem and network permissions.

`entroly doctor --privacy` inspects selected source patterns and configuration.
It does not observe all network traffic, verify every integration, audit OS ACLs,
or certify that no data can leave the machine. Verification and secret-scanning
heuristics can miss cases; neither guarantees no hallucinations or full redaction.

## Contact

Use [SECURITY.md](SECURITY.md) for private vulnerability or sensitive privacy
reports. Use [GitHub Issues](https://github.com/juyterman1000/entroly/issues) for
non-sensitive questions. Never attach private code, credentials, or unredacted
recovery artifacts to a public report.
