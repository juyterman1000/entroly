# Entroly changelog

This file is the stable entry point for user-facing release history. Published
version-specific notes live in [GitHub Releases](https://github.com/juyterman1000/entroly/releases).

## Unreleased

- With session rescue off, shell tool output of at most 4,000 characters is
  forwarded verbatim (`ENTROLY_SHELL_VERBATIM_CHARS`, `ENTROLY_SHELL_TOOLS`);
  a 30-file `ls -l` had been rewritten with sizes rounded to "1KB" and modes
  and timestamps dropped. Tool results are now matched to their tool's name
  through the call id, so `ENTROLY_TOOL_RESULT_EXCLUDE` also protects
  Anthropic tool results, which it never matched before.
- The templated log form no longer factors out values that contain its `|`
  column separator. A value like `tags=a|b` split into two column entries,
  shifting every later entry so each line's values sat beside another line's
  (which user had which tags was wrong, though every value was present). Those
  lines now stay verbatim. New tests check that every compressed form keeps
  paired facts -- test and outcome, file:line and error code, row id and
  value -- together.
- Traffic Receipts now carry a conservative net saving
  (`net_saving_conservative_micro_usd`) for priced Claude models, on by
  default. It prices only what Entroly changed: removals at the cheapest
  possible rate, additions and any cache prefix Entroly broke at the most
  expensive, scaled by the provider's own token count. Unchanged requests are
  zero, the provider's cache discount is never credited, values can be
  negative, and it is never added to the headline. Uses a bundled, dated
  Anthropic price table unless `ENTROLY_PRICING_CATALOG` is set
  (`ENTROLY_BUNDLED_PRICING=0` disables it). Receipts now share the
  `X-Entroly-Request-Id` the client sees when the client sends no request id.
- A tool result cut to a partial codec view (for example a code skeleton)
  now opens with a one-line notice naming the view and how to get the exact
  content; bodies were previously dropped with no marker on the
  `ENTROLY_SESSION_RESCUE=0` and OpenClaw paths.
- Provider TLS now trusts the operating-system certificate store plus the
  bundled roots, in the Python proxy and `entroly-rs`, so Entroly works behind
  TLS-inspecting corporate gateways. `NODE_EXTRA_CA_CERTS` adds roots instead
  of replacing the store; `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` still name the
  whole trust set. `ENTROLY_TLS_TRUST=bundled` restores bundled roots. Adds the
  `truststore` dependency. See
  [Troubleshooting](docs/TROUBLESHOOTING.md#certificate_verify_failed-or-unknownissuer-behind-a-corporate-network).
- Claude Fable 5.1, Opus 5.5, Opus 5, Sonnet 5.5, Sonnet 5 and Haiku 5.5 now
  resolve to their verified 1M-token window and published prices, and Opus 4.6
  to its 1M window; they had fallen back to 128K, so session rescue compacted
  history at ~113K tokens. Savings are priced at Opus 4.5-4.8's $5/$25 and
  Haiku 4.5's $1/$5, with Bedrock and router model names recognised.
  Injected repository context is budgeted against at most a 200K window
  (`ENTROLY_INJECTION_REFERENCE_WINDOW`, 0 disables), which lowers GPT-5.6
  and Gemini 3.6 injection ceilings from 256K to 60K.
- Conversation compression (`ENTROLY_SESSION_RESCUE=0`) no longer rewrites
  Anthropic `thinking`/`redacted_thinking` blocks, which the API rejects when
  modified, and measures utilisation against the request model's window
  instead of a fixed 128K.
- Corrected optional WITNESS calibration's lower-tail rank and empty-evidence
  handling. Legacy or invalid stored profiles now return all-SUPPRESS thresholds
  rather than reactivating permissive learned values; recalibrate explicitly.
  Missing profiles retain uncalibrated defaults. See the
  [assumptions, scope and compatibility notes](docs/witness-calibration.md).
- Newly prepared contexts sign a full Context Commit anchor so exact recovery
  and proof-guided planning can reach committed omissions beyond the 20-item
  receipt preview. Older preparations remain listed-only; re-prepare to opt in.
  Audit loading now verifies the exact payload it consumes. See
  [recovery compatibility and rollback](docs/verified-efficiency-layer.md#recovery-compatibility-and-rollback).
- Proof-guided runtime v2 rejects replies that do not carry the issued request's
  `revision` and `request_sha256`, and serializes cooperating local writers.
  CLI, MCP and HTTP clients must retain that binding beside each model call.
  Existing v1 sessions are inspect-only: re-prepare with a new idempotency key
  and upgrade all writers together. See the
  [protocol and rollback limits](docs/proof-guided-context-fixed-point.md#durable-host-driven-protocol).
- Runtime native-engine repair now requires explicit consent. Install
  `entroly-core` yourself, call `entroly.repair()`, or opt in to startup repair
  with `ENTROLY_ENABLE_SELF_HEAL=1`. `ENTROLY_NO_SELF_HEAL=1` and
  `ENTROLY_AIR_GAP=1` override consent. Missing-native diagnostics remain visible.
- Corrected the PEP 668 marker path so repair respects externally managed Python.
- Context Receipt and Context Commit file writes use atomic replacement and
  private POSIX file modes. Windows inherits directory ACLs; existing directory
  permissions and unencrypted storage remain operator responsibilities.
- Corrected privacy/telemetry disclosures and removed unsupported accuracy and
  unnamed-product comparison claims. Privacy diagnostics no longer certify the
  absence of network traffic. Historical loss cases remain documented.

These changes are not part of the published releases below until a patch release
is completed. See [recovery-data security](docs/recovery-data-security.md).

## Current release

- [`v1.0.84`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.84)

## Recent releases

- [`v1.0.83`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.83)
- [`v1.0.82`](https://github.com/juyterman1000/entroly/releases/tag/entroly-v1.0.82)

## Changelog contract

A release note must describe user-visible behavior rather than only internal
implementation work. Include, where applicable:

- capabilities added, changed, or removed;
- installation and upgrade impact;
- compatibility and migration requirements;
- security, privacy, persistence, or network-boundary changes;
- fixes for user-visible failures;
- verification commands and linked evidence;
- known limitations and regressions;
- rollback or uninstall guidance.

Quantitative statements must identify the exact version, workload, model or
provider when applicable, token or context budget, sample, protocol, and raw
artifact. A result from an earlier version is not automatically evidence for a
new release.

## Publication rule

Do not add a release as current until its tag and coordinated artifacts are
public and resolve to the intended commit. The canonical release procedure is
[`docs/RELEASE.md`](docs/RELEASE.md), and the coordinated publication workflow is
[`.github/workflows/entroly-publish.yml`](.github/workflows/entroly-publish.yml).
