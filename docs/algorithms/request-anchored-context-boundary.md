# Request-anchored context boundary

Status: active in the non-streaming and streaming proxy paths. This is a
recoverable history-pruning guard, not a claim that an upstream provider will
always accept the result or that a reduced history preserves answer quality.

## Invariant

For a request containing prior user turns, partition the provider-native
sequence into an immutable prefix, complete historical turns, and the current
turn. A proposal may omit only a prefix of complete historical turns. The
current turn and all remaining provider-native items are copied exactly.
Tool results in an Anthropic or Gemini user-role item remain attached to the
assistant/model call that produced them. OpenAI-style tool messages remain in
their enclosing user turn.

For each proposed boundary, Entroly builds the **final provider payload**,
including the recovery notice, and estimates its tokens. It accepts the
proposal only if that final estimate fits the configured ceiling and the
omitted items can be read back through the public `entroly recover` store.
The recovery digest covers the serialized original provider items; it is a
content digest, not a Merkle proof.

If the current turn itself is too large, the schema is ambiguous, the request
contains media whose token cost cannot be estimated from its URL, or local
recovery fails, the guard forwards the original payload. An upstream context
error may still occur. Silent deletion of the active request is not an
acceptable fallback.

## Wire formats

The boundary follows the request format rather than a model-brand allowlist:

| Request shape | Recovery notice | History field |
| --- | --- | --- |
| Chat Completions and compatible APIs | system message | `messages` |
| Responses API | append to top-level `instructions` | `input` |
| Anthropic Messages | append to top-level `system` | `messages` |
| Gemini GenerateContent | append to top-level `systemInstruction.parts` | `contents` |

DeepSeek, Mistral, GLM, Kimi, Ollama, OpenRouter, and custom endpoints can use
the Chat Completions-compatible shape without model-specific guard code.
Entroly's current direct proxy transports are OpenAI-compatible, Anthropic,
and Gemini. A provider using a different native request protocol needs a
transport and wire-format adapter before Entroly can safely rewrite it.

## Evidence boundary

The local estimate serializes the whole request, including tools and system
content, then counts with Entroly's local tokenizer or character heuristic.
It is not the provider's exact tokenizer. The proxy uses an 85% window
ceiling as a margin and recognizes an upstream overflow response for one
additional recoverable retry. The guard does not claim a 413/400 cannot occur.

This deterministic boundary solves a transport integrity problem. Whether
skeletal context or any other reduced representation improves a task is a
separate question. The [evidence-adaptive granularity frontier](evidence-adaptive-granularity.md)
requires matched baseline and candidate outcomes before promoting a context
resolution for an exact task/model/tokenizer scope. It is not automatically
enabled for public traffic without that evidence.
