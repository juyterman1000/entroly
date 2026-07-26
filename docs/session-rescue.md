# Runaway-session rescue and prompt-cache safety

Entroly's session rescue is an outbound safety controller inside
`entroly proxy`. It protects a provider request without mutating the transcript
stored by Claude Code, Codex, OpenClaw, an IDE, or another agent host.

## Runtime contract

For each supported request, the proxy:

1. detects and compresses heavy textual tool output in the live zone;
2. writes one exact full-original recovery span before changing that block;
3. estimates the resulting active context against the resolved model window;
4. defers optional reshaping at the soft watermark when provider usage has
   reported a warm cache;
5. overrides that deferral for a detected loop or the hard watermark;
6. preserves the configured recent tail and all ordinary user messages;
7. returns `413 session_context_rescue_required` when the request remains above
   the failure watermark instead of forwarding a likely provider rejection.

The controller is request-driven because Entroly does not own an agent host's
saved transcript. It runs automatically on every request routed through the
long-lived proxy daemon; it cannot rescue traffic that bypasses that proxy.

## Cache behavior

Provider caches require reusable prefixes. Entroly therefore uses two distinct
zones:

```text
stable provider system/history | current raw turn | dynamic Entroly context
```

Changing Entroly evidence is appended after the newest safe user/tool content,
not prepended to the system prompt. On the next turn, the provider can still
match the historical prefix up to the previous live-zone boundary. Old tool
output is compressed deterministically without using the latest query, so the
same raw block produces the same forwarded bytes.

The cache aligner accepts only an exact SHA-256 match. It never substitutes an
older context because the token sets merely look similar. A provider cache
lease is based on provider usage fields, not Entroly's internal match counter.

This layout improves the chance of a cache hit; it does not guarantee one.
Model thresholds, TTLs, routing, account terms, and provider implementation can
all affect the result. Inspect provider usage metadata and invoices.

## Recovery

The default recovery store is:

```text
${ENTROLY_DIR:-~/.entroly}/session_rescue_recovery.json
```

A compressed block contains a marker such as:

```text
[entroly-recovery:<receipt-id>:<span-id>]
```

The span is the complete original textual tool block. Retrieve it
programmatically:

```python
from entroly import CompressionRetrievalStore

store = CompressionRetrievalStore(
    ".entroly/session_rescue_recovery.json"
)
span = store.get_span(receipt_id, span_id)
print(span.content)
```

Or expose the focused local MCP server:

```bash
export ENTROLY_COMPRESSION_STORE=.entroly/session_rescue_recovery.json
entroly-compression-mcp
```

Its tools include `retrieve_compressed_span`, `search_compressed_spans`, and
`list_compression_receipts`.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `ENTROLY_SESSION_RESCUE` | `1` | Enable the proxy guard |
| `ENTROLY_SESSION_RESCUE_STORE` | under `ENTROLY_DIR` | Exact recovery-store path |
| `ENTROLY_SESSION_RESCUE_STORE_MAX_BYTES` | `536870912` | Fail before mutation if the store would exceed 512 MiB |
| `ENTROLY_SESSION_SOFT_WATERMARK` | `0.70` | Begin normal high-water rescue |
| `ENTROLY_SESSION_HARD_WATERMARK` | `0.88` | Override cache deferral |
| `ENTROLY_SESSION_TARGET_WATERMARK` | `0.62` | Target after rescue |
| `ENTROLY_SESSION_FAILURE_WATERMARK` | `0.98` | Refuse unsafe forward |
| `ENTROLY_SESSION_LOOP_MIN_WATERMARK` | `0.40` | Minimum pressure for loop rescue |
| `ENTROLY_SESSION_TAIL_MESSAGES` | `8` | Recent messages that are not compacted |
| `ENTROLY_SESSION_TOOL_BUDGET` | `1200` | Per-block live-zone token budget |
| `ENTROLY_CACHE_STABLE_INJECTION` | `1` | Put dynamic context after stable history |

Watermarks must satisfy:

```text
loop_min <= target < soft < hard < failure
```

Invalid configuration is reported at startup. If the recovery store cannot be
initialized, recoverable rescue is disabled with an explicit error rather than
silently pretending it is active.

## Response signals

Useful response headers include:

```text
X-Entroly-Session-Rescue
X-Entroly-Session-Original-Tokens
X-Entroly-Session-Forwarded-Tokens
X-Entroly-Session-Tokens-Saved
X-Entroly-Session-Stable-Prefix-Messages
X-Entroly-Session-Recovery-Receipts
X-Entroly-Context-Injection
x-entroly-compression-mode
x-entroly-compressed-blocks
```

`GET /stats` reports controller rescues, cache deferrals, blocks, failures, and
the last decision. These are local operational signals, not provider billing
proof.

## Security and limitations

- The JSON recovery store contains raw original tool output. Protect it like
  build logs or source code: restrict filesystem access, exclude it from source
  control, and apply an appropriate retention policy.
- The current token estimator is deterministic but approximate. The provider's
  tokenizer and usage metadata are authoritative.
- User messages, multimodal blobs, unknown tool-result schemas, and Gemini
  thought signatures are never guessed at or destructively compacted.
- A `413` can still require starting a fresh turn or retrieving exact evidence
  into a smaller prompt. Entroly does not claim that every overflow can be
  compressed safely.
- Bypass mode intentionally leaves the request unchanged except for separately
  configured outbound security policy.

## Verification

```bash
python -m pytest \
  tests/test_session_rescue.py \
  tests/test_proxy_session_rescue.py \
  tests/test_cache_stable_live_zone.py \
  tests/test_compression_proxy.py
```

The tests cover exact recovery, byte-stable frozen history, loop and hard-water
activation, cache deferral, explicit overflow blocking, provider message
shapes, and preservation of Gemini identifiers and thought signatures.
