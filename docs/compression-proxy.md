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

With session rescue turned off (`ENTROLY_SESSION_RESCUE=0`), the proxy's
pattern-based tool compression applies two name-based rules. A tool result is
matched to its tool's name through the call that produced it (Anthropic
`tool_use_id`, OpenAI `tool_call_id`):

- Shell output (`Bash`, `shell`, `run_terminal_cmd`, `execute_command` and
  similar; add names with `ENTROLY_SHELL_TOOLS`) of at most 4,000 characters
  is forwarded exactly as printed, so a short `ls -l` or `git status` keeps
  its sizes, modes, timestamps and paths. `ENTROLY_SHELL_VERBATIM_CHARS` sets
  the floor; `0` turns it off.
- Tools named in `ENTROLY_TOOL_RESULT_EXCLUDE` are never compressed.

On the default path (session rescue on) the live proxy applies a tool-result
policy before the lossy codec runs:

- **File reads and program source are forwarded as sent.** A result is
  protected when its tool is a file reader (`Read`, `read_file`, `view`,
  `cat`, `get_file_contents`, `str_replace_editor` and similar, matched on the
  last segment of the name so `mcp__filesystem__read_file` counts), or when
  its text is a numbered file view or reads as source code. The line codec
  keeps anchors and drops the lines between them; in source those lines are
  the function body. Set `ENTROLY_PROTECT_FILE_READS=0` to restore the earlier
  behaviour, where every tool result over the block budget is cut.
- **Tools named in `ENTROLY_TOOL_RESULT_EXCLUDE` are never compressed**, on
  this path as well as the pattern-based one.
- **A byte-identical repeat becomes a pointer.** When a tool result is the
  same bytes as an earlier one in the same request (a test run twice, a file
  read again), the later copy is replaced by one line naming the earlier
  call. The earlier copy is forwarded untouched, so nothing is lost. Results
  under 1,024 characters are left alone. `ENTROLY_TOOL_RESULT_DEDUP=0` turns
  it off.
- **A rewrite must remove at least 20% of a block.** A block just over the
  budget used to lose a few lines for a few tokens; a rewrite that keeps more
  than 80% costs more than it saves on the turn it is made, because the
  rewritten block is written to the provider cache at 1.25x the input rate.

The response reports what happened in `X-Entroly-Protected-Blocks` and
`X-Entroly-Deduped-Blocks` beside the existing compression headers.

`compress_proxy_payload()` called directly keeps its earlier behaviour unless
a policy is passed:

```python
from entroly.compression_proxy import ToolResultPolicy, compress_proxy_payload

result = compress_proxy_payload(body, provider="anthropic",
                                policy=ToolResultPolicy.from_names("my_tool"))
```

## Repository context is opt-in

The proxy forwards a request with nothing added unless
`ENTROLY_CONTEXT_INJECTION` says otherwise:

| Value | Requests that receive selected repository context |
|---|---|
| `off` (default) | none |
| `chat` | requests that declare no tools |
| `always` | every request whose newest item is a new user question |

Injected context is text the client did not send, so it can only make a
request larger, and selecting it costs seconds per question on a large index.
It is worth that when the model has no other way to see the code. It is pure
cost for an agent that reads files with its own tools, or for an application
whose prompts are not about the repository the proxy was started in, and the
proxy cannot tell those apart from the request. In `chat` and `always` mode
selection runs once per new user question, never on a tool turn; the reason a
request received no context is reported in `X-Entroly-Context-Injection`
(`off`, `agent-has-tools`, `not-a-new-question`).

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

## Headers a gateway behind Entroly depends on

The proxy forwards a provider's own header families (`anthropic-*`,
`openai-*`, `x-stainless-*`, tracing headers, the provider's auth header) and
drops every other client header, so an unknown header never reaches a
provider by accident.

That default also removes the tags a gateway *behind* Entroly keys on: a run
or team id used for usage attribution, or a trace id for an observability
layer. Name the ones to keep:

```bash
ENTROLY_FORWARD_HEADERS="x-ccb-run-id, x-team-*" entroly proxy
```

- Entries are exact header names or `prefix-*` (at least three characters).
- Off by default. Nothing extra is forwarded unless it is listed.
- Credentials (`authorization`, `x-api-key`, `api-key`, `x-goog-api-key`),
  cookies, transport headers and Entroly's own `x-entroly-*` headers are
  refused even if listed, including through a prefix that would match them.
  Refused entries are named in a warning; accepted ones in an info line.
- The same list applies to Anthropic, OpenAI and Gemini upstreams.

