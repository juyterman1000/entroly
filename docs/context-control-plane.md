# Context control plane

Entroly is the context control plane around an existing agent runtime. Claude Code, Codex, or OpenClaw still owns the conversation and executes the agent; Entroly selects, remembers, verifies, and proves the context delivered to it.

For OpenClaw, this boundary is provider-independent. OpenClaw resolves the
provider, model, authentication, failover, normalized messages, and prompt
budget. Entroly transforms only eligible unsigned text before each model call.
Signed text, thinking blocks, images, tool calls, tool-result identifiers and
metadata, provider signatures, message metadata, and recent turns remain
exact. Eligible unsigned text inside an older tool result may be compressed. If
OpenClaw cannot provide a finite budget and the operator has not configured a
fallback, Entroly returns the original context with an actionable diagnostic.

OpenClaw receipts do not persist the plaintext request, matched query terms, or
reversible per-term digests; they retain only lengths and counts. Append-only
proposals have unique nonces and immutable payload digests. A proposed receipt
becomes accepted only after the plugin validates its message structure and
reveals a precommitted 256-bit challenge. Entroly persists an HMAC signature
from a private key outside the workspace receipt, never the challenge secret.
All content digests are keyed too, so receipts do not become offline prompt
guessing oracles. That verifiable state proves context-engine validation, not
delivery by the downstream model provider.
Default workspace stores reject symlink escapes, atomically persist private
files, and reserve the same byte count before and after acceptance. A hard
file/byte quota refuses new writes without deleting old audit history.

## Model metadata with explicit trust

The bundled registry records context limits, output limits, prices, capabilities, aliases, and a trust state. Verified entries are backed by first-party provider documentation. Announced or discovered entries remain visibly labelled and may omit fields that have not been verified.

Remote discovery is opt-in:

```bash
python -m entroly.models discover openrouter
```

Set `OPENROUTER_API_KEY` for OpenRouter discovery. The credential is used only for that request and is never written to the registry. Local Ollama and LM Studio discovery remain available without a remote credential. Invalid or incomplete provider metadata is rejected or retained as unknown instead of being guessed.

## Scoped, revocable attachment

Create a four-hour, project-bound grant and install it in a supported client:

```bash
entroly attach create --client claude --project . --ttl 4h --install
entroly attach create --client codex --project . --ttl 4h --install
entroly attach create --client openclaw --project . --ttl 4h --install
```

Each grant has least-privilege tool scopes, an expiry, and a high-entropy bearer token stored only in a permission-restricted local token file. The database stores only its SHA-256 hash. Raw tokens are not placed in client configuration or process arguments.

Every MCP tool call re-checks the token, grant, expiry, revocation state, tool scope, and project binding. Unscoped resources, prompts, and templates are removed from an attached server. If client installation fails, Entroly removes the partial client entry when possible and revokes the grant.

```bash
entroly attach list
entroly attach revoke <grant-id> --uninstall
```

Revocation is applied before client configuration removal, so access is denied even if the client cannot be cleaned up automatically. The error includes the manual recovery command.

## Reliable event delivery

Slack, Discord, and Telegram gateways persist events before delivery. Events have idempotency keys, destination-aware ordering, retry leases, exponential backoff with jitter, token redaction, and dead-letter state.

Startup synchronously replays due events before announcing that a gateway is online. A restart therefore cannot silently skip an accepted event. Remaining queued or dead-letter events produce visible diagnostics instead of a false success signal.

## Context-oriented sessions

The dashboard is an evidence viewer rather than another chat UI. It provides:

- a searchable session list;
- a selected-versus-omitted context ring;
- receipt and chain-integrity status;
- bounded evidence excerpts and an omitted-evidence explorer;
- model, context, output, and cost information when the receipt contains enough data.

Unknown values stay `Unknown`; Entroly does not invent usage or cost. Cost estimates identify their basis and exclude cache discounts, long-context tiers, provider routing, and negotiated pricing.

The scanner is read-only and bounded by file count, file size, directory depth, and excerpt size. Corrupt, conflicting, oversized, or incomplete artifacts appear as diagnostics.

## Recovery guarantees

| Failure | Predictable outcome |
|---|---|
| Attachment expires or is revoked | The next tool call is denied and audited |
| Client install partially fails | Grant is revoked; automatic cleanup is attempted; manual recovery is printed if needed |
| Gateway process restarts | Due persisted events replay before the online announcement |
| Provider metadata is malformed | Bad values are rejected or kept unknown with warnings |
| Receipt or session chain is corrupt | Dashboard reports the integrity problem without mutating source data |
| Pricing or usage is incomplete | UI reports `Unknown`, not a fabricated estimate |

Entroly does not silently discard accepted work, weaken an attachment scope after creation, or claim stronger verification than its stored evidence supports.
