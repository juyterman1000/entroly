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

## Live proxy mode settings

The provider-light runtime helper reads environment settings:

```bash
export ENTROLY_COMPRESSION_PROXY_MODE=elc
export ENTROLY_ELC_BUDGET_TOKENS=1200
export ENTROLY_COMPRESSION_STORE=.entroly/compression-store.json
```

Then:

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
pytest tests/test_compression_proxy.py tests/test_compression_proxy_scoreboard.py tests/test_compression_retrieval_store.py -v
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

The next product milestone is wiring `compress_proxy_payload_from_env()` directly
into the HTTP proxy request path behind `ENTROLY_COMPRESSION_PROXY_MODE=elc`.
