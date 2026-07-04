# Entroly Competitive Gap Plan

Goal: make Entroly the most trusted local context OS for AI coding agents by
winning on first-run success, recoverability, verification, and reproducible
token savings.

## Honest current position

The market rewards products that feel polished immediately:

- one-command setup,
- agent wrapping,
- dashboard,
- public benchmark story,
- native speed,
- exact recovery markers.

Entroly's strongest differentiator is not blind compression. It is auditable,
evidence-preserving context control:

- Evidence-Locked Compression (ELC),
- Context Receipts,
- WITNESS / EICV verification path,
- proxy control plane,
- Memory Fabric.

## Winning thesis

Entroly should compress context with evidence locks, receipts, retrieval, and
verification.

That means Entroly should be judged not only by token savings, but by:

1. Did we preserve the answer-critical evidence?
2. Did we emit a receipt explaining what was omitted?
3. Can omitted spans be retrieved later?
4. Can the final answer be verified against retained evidence?
5. Did we save enough tokens to materially reduce the API bill?

## What landed in 1.0.28

- `entroly.evidence_locked_compression`
- `entroly.compression_proxy`
- `benchmarks/compression_proxy_scoreboard.py`
- `tests/test_compression_proxy.py`
- `tests/test_compression_proxy_scoreboard.py`

The scoreboard covers:

- 100k-style build logs,
- deep JSON arrays,
- SRE incident logs.

Each scenario must satisfy:

- evidence preserved,
- receipt emitted,
- useful savings,
- aggregate mean savings >= 70%.

Run:

```bash
python benchmarks/compression_proxy_scoreboard.py --json
pytest tests/test_compression_proxy.py tests/test_compression_proxy_scoreboard.py -v
```

## Next milestones

### 1. Live proxy flag

Add:

```bash
ENTROLY_COMPRESSION_PROXY_MODE=elc
```

and route existing proxy tool-output compression through
`compress_proxy_payload()`.

### 2. Omitted-span retrieval store

ELC currently emits recoverable span receipts. Add a local retrieval store so
large omitted spans can be fetched by hash or span id.

Target API:

```python
store.get(receipt_id, span_id)
```

Target MCP tool:

```text
entroly_retrieve_compressed_span
```

### 3. Proxy headers

Expose:

```text
x-entroly-compression-mode
x-entroly-original-tokens
x-entroly-compressed-tokens
x-entroly-tokens-saved
x-entroly-savings-ratio
x-entroly-compressed-blocks
```

The `ProxyCompressionResult.headers()` surface already returns these.

### 4. Verification loop

Add:

```text
ELC -> compressed prompt -> answer -> WITNESS/EICV -> retrieve omitted span on failure -> retry once
```

This is the margin Entroly should own: compression plus evidence verification.

### 5. Rust fast path

Move ELC scoring to `entroly-core` once the Python semantics are stable.

Target:

```text
Rust ELC for speed, Python ELC as audited reference implementation.
```

### 6. Public benchmark parity page

Publish benchmarks across:

- JSON arrays,
- build logs,
- shell output,
- SRE incident debugging,
- codebase exploration,
- grep/search results,
- prompt-cache stability,
- output token shaping.

Do not claim victory without reproducible numbers.

## Public claim allowed now

```text
Entroly adds Evidence-Locked Compression: a proxy-ready compression surface that
preserves failure evidence, query matches, JSON outliers, omitted-span receipts,
and measurable savings on heavy tool payloads.
```

## Public claim not allowed yet

```text
Entroly beats every context optimizer across every benchmark.
```

That claim requires reproducible third-party parity benchmarks and live proxy
integration.
