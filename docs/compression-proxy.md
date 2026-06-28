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

Set these before starting the Entroly proxy:

```bash
export ENTROLY_COMPRESSION_PROXY_MODE=elc
export ENTROLY_ELC_BUDGET_TOKENS=1200
export ENTROLY_COMPRESSION_STORE=.entroly/compression-store.json
entroly proxy
```

When `ENTROLY_COMPRESSION_PROXY_MODE=elc`, Entroly installs the live proxy hook
at package import time. The hook replaces the existing tool-output compression
function with the Evidence-Locked Compression proxy surface. If the env var is
absent or different, nothing changes.

Programmatic helper:

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
- OpenAI Responses-style tool/text blocks when enabled.

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

When a `CompressionRetrievalStore` is supplied, omitted spans are stored locally.
The receipt includes:

```json
{
  "retrieval": {
    "receipt_id": "...",
    "span_count": 3,
    "span_ids": ["..."]
  }
}
```

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

## Positioning against Headroom

Headroom's strongest idea is reversible compression through CCR. Entroly now has
the equivalent recoverability foundation with a different thesis:

```text
Headroom retrieves dropped context.
Entroly retrieves dropped context and keeps auditable evidence receipts for what was compressed, why it was compressed, and how to recover it.
```

That is the product wedge: Entroly is not only a compression proxy; it is an
auditable, recoverable evidence-control plane for compressed LLM context.
