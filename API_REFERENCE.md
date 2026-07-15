# API Reference

This page documents the stable, high-level Python and CLI entry points. Entroly
also exposes advanced modules; importing internal modules directly does not make
them part of the compatibility contract.

## Installation

```bash
python -m pip install entroly
```

Add `entroly[native]`, `entroly[proxy]`, or `entroly[full]` only when the
corresponding capability is required. See
[SUPPORTED_VERSIONS.md](SUPPORTED_VERSIONS.md).

## `compress`

```python
compress(
    content: str,
    budget: int | None = None,
    content_type: str | None = None,
    target_ratio: float = 0.3,
    profile: str | None = None,
) -> str
```

Performs query-agnostic structural compaction. `budget` is an estimated-token
upper bound and overrides `target_ratio`. This API cannot promise retention of
the answer to an unknown future question; use `optimize` when a task query is
available. Non-empty input with a positive budget must not silently become an
empty result.

## `compress_messages`

```python
compress_messages(
    messages: list[dict[str, Any]],
    budget: int = 50_000,
    preserve_last_n: int = 4,
    model: str | None = None,
    client_key: str | None = None,
    distill: bool = True,
    profile: str = "balanced",
    target_ratio: float | None = None,
) -> list[dict[str, Any]]
```

Fits a conversation list within a budget while preserving recent messages and
progressively compacting older content. `model` can clamp the budget only when
model metadata has sufficient trust. Callers should retain the original message
list until the provider request succeeds.

## `optimize`

```python
optimize(
    fragments: list[dict[str, Any]],
    budget: int = 128_000,
    query: str = "",
) -> dict[str, Any]
```

Selects task-conditioned context. Each fragment requires `content` and `source`;
`token_count` is optional. The result includes `selected`, `total_tokens`,
`fragments_selected`, `fragments_total`, and `context_text`.

## `verify`

```python
verify(code: str, context: str = "") -> dict[str, Any]
```

Checks identifiers in generated code against supplied context. The result
includes `ipd`, `verdict`, counts, and ungrounded identifiers. It is evidence for
review, not proof that code is correct or safe.

## `detect_hallucination`

```python
detect_hallucination(
    response: str,
    context: str = "",
    prompt: str = "",
) -> dict[str, Any]
```

Runs local WITNESS-based answer analysis and labelled auxiliary diagnostics.
The result includes risk, verdict, recommendation, primary signal, individual
signals, and flagged claims. Consult the current benchmark artifact and
limitations before choosing a production threshold.

## Context Receipts

```python
create_context_receipt(
    documents,
    query: str,
    budget: int = 8_000,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
    recoverable: bool = False,
) -> dict[str, Any]

context_receipt_from_path(
    path: str,
    query: str,
    budget: int = 8_000,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
) -> dict[str, Any]

render_context_receipt(receipt, prefer_rust=True) -> str
explain_receipt_omission(receipt, chunk_id, prefer_rust=True) -> str
recover_receipt_omission(receipt, chunk_id=None, *, store_dir=None) -> list[dict]
```

`documents` may be a `path -> text` mapping, `(path, text)` tuples, or mappings
with `source_path` and `text`. Exact omitted-content recovery requires
`recoverable=True` and the matching local store.

## Context Commits

```python
create_context_commit(
    documents,
    *,
    query: str,
    token_budget: int,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    parent_commit_id: str | None = None,
    prefer_rust: bool = True,
) -> dict
verify_context_commit(commit) -> ContextCommitVerification
replay_context(commit) -> list[dict[str, Any]]
```

Context Commits are self-contained and may include source text. Verification
checks integrity and contract consistency; authenticated custody requires the
optional signing/attestation APIs. See [docs/context-commits.md](docs/context-commits.md).

## Memory OS

`MemoryOS` is the dependency-light facade for adding, selecting, and auditing
local memory context. Use the runnable examples in
[`examples/memory_os_e2e_demo.py`](examples/memory_os_e2e_demo.py) and the
product-surface map for advanced memory APIs.

## CLI

Run `entroly --help` for the installed version's authoritative command list and
`entroly <command> --help` for options. Primary command groups:

| Goal | Commands |
| --- | --- |
| Prove installation | `doctor`, `verify-claims`, `simulate`, `perf` |
| Integrate | `init`, `attach`, `serve`, `proxy`, `wrap`, `go` |
| Select and audit | `optimize`, `ingest`, `select`, `receipt`, `context-commit`, `audit`, `explain` |
| Verify | `verify`, `verify-code`, `witness`, `ravs` |
| Operate | `status`, `dashboard`, `daemon`, `config`, `telemetry`, `cache` |
| Manage state | `export`, `import`, `migrate`, `clean`, `sync` |

## MCP and HTTP schemas

The installed MCP server is the authoritative tool-schema source. Register the
stdio command, list tools through the client, and use an explicit allowlist for
security-sensitive integrations. HTTP proxy behavior is documented in
[docs/compression-proxy.md](docs/compression-proxy.md); provider protocol,
streaming, tools, and status codes remain authoritative.

## Stability

Public imports documented here follow semantic versioning. Receipt and Context
Commit schemas are additive within 1.x. Experimental modules, benchmark scripts,
and symbols not documented here may change without the same deprecation window.
