# Evidence-Locked Compression Proxy

Entroly's compression proxy mode is designed to beat blind compression systems by
being evidence-first and recoverable.

Core principle:

```text
Compress aggressively around evidence, never through evidence.
```

## Public API

```python
from entroly import compress_proxy_payload, CompressionRetrievalStore

store = CompressionRetrievalStore(".entroly/compression-store.json")

result = compress_proxy_payload(
    body,
    provider="anthropic",
    query="why did CI fail?",
    budget_tokens=1200,
    retrieval_store=store,
)

forward_body = result.body
headers = result.headers()
receipt = result.receipt.as_dict()
```

## Live HTTP proxy mode

The live proxy enables recoverable session rescue by default:

```bash
export ENTROLY_SESSION_RESCUE_STORE=.entroly/session_rescue_recovery.json
export ENTROLY_SESSION_TOOL_BUDGET=1200
entroly proxy
```

On every supported outbound request, the proxy compresses eligible textual tool
output, persists one exact full-original recovery span, evaluates the active
context against the model window, and either forwards, rescues, or returns an
actionable capacity error. See [session rescue](session-rescue.md).

The separate environment-driven programmatic helper remains opt-in:

```python
from entroly import compress_proxy_payload_from_env

result = compress_proxy_payload_from_env(body, provider="openai", query="why did build fail?")
```

Defaults are safe: if `ENTROLY_COMPRESSION_PROXY_MODE` is not `elc`, the helper
passes requests through unchanged.

## What gets compressed

By default Entroly compresses:

- OpenAI `role=tool` messages,
- OpenAI `role=function` messages,
- Anthropic `tool_result` blocks,
- OpenAI Responses-style tool/text blocks when enabled,
- known textual fields in Gemini `functionResponse` and
  `codeExecutionResult` parts.

Gemini function-call IDs, part order, thought signatures, and unknown
structured fields are preserved. Ambiguous structures pass through unchanged.

Entroly preserves user and assistant text by default because the latest user
message is usually the semantic target. User-message compression is explicit:

```python
compress_proxy_payload(body, compress_user_messages=True)
```

or:

```bash
export ENTROLY_ELC_COMPRESS_USER=1
```

## Receipts and retrieval

Every compressed block can emit:

- original tokens,
- compressed tokens,
- savings ratio,
- preserved anchor counts,
- omitted spans,
- recoverability metadata.

When a `CompressionRetrievalStore` is supplied, one span containing the complete
original textual block is stored locally before the forwarded copy is changed.
This is stronger than depending on the compressor's bounded diagnostic
omission list. The receipt includes:

```json
{
  "retrieval": {
    "receipt_id": "...",
    "span_count": 1,
    "span_ids": ["..."]
  }
}
```

The forwarded block also contains
`[entroly-recovery:<receipt-id>:<span-id>]`, so an agent can request the exact
original without relying on response headers.

Fetch a span:

```python
span = store.get_span(receipt_id, span_id)
print(span.content)
```

Search omitted spans:

```python
matches = store.search("auth timeout")
```

## MCP retrieval server

Entroly also ships a focused MCP server for omitted-span retrieval:

```bash
export ENTROLY_COMPRESSION_STORE=.entroly/compression-store.json
entroly-compression-mcp
```

It exposes:

```text
retrieve_compressed_span(receipt_id, span_id)
search_compressed_spans(query, limit=5)
list_compression_receipts()
```

Use it when a compressed prompt contains a retrieval receipt and the agent needs
more exact context.

## Proxy headers

`ProxyCompressionResult.headers()` returns:

```text
x-entroly-compression-mode
x-entroly-original-tokens
x-entroly-compressed-tokens
x-entroly-tokens-saved
x-entroly-savings-ratio
x-entroly-compressed-blocks
X-Entroly-Session-Rescue
X-Entroly-Session-Recovery-Receipts
X-Entroly-Context-Injection
```

## Benchmark gate

Run:

```bash
python benchmarks/compression_proxy_scoreboard.py --json
pytest tests/test_compression_proxy.py tests/test_compression_proxy_scoreboard.py tests/test_compression_retrieval_store.py tests/test_compression_proxy_live.py -v
```

The scoreboard requires:

- answer-critical evidence preserved,
- receipts emitted,
- mean savings >= 70%,
- local deterministic execution.

## Product positioning

Entroly's compression proxy is built around recoverability and auditability, not
blind shrinking:

```text
Entroly retrieves dropped context and keeps auditable evidence receipts for what was compressed, why it was compressed, and how to recover it.
```

That is the product wedge: Entroly is not only a compression proxy; it is an
auditable, recoverable evidence-control plane for compressed LLM context.
