# Context Assurance across agent runtimes

Entroly supports Hermes Agent, OpenClaw, and OpenCode through their native
extension points. The integration boundary is deliberately small: each host keeps
ownership of its models, provider credentials, transcript, retries, and tool
execution; Entroly controls context selection, exact recovery, receipts, and
verification.

| Runtime | Native integration | Entroly responsibility | Host responsibility |
|---|---|---|---|
| Hermes Agent | `ContextEngine` plugin | request-only selection, compaction, exact `ccr:` replay tool, status and usage observation | provider routing, transcript, tool loop |
| OpenClaw | context-engine plugin + local JSONL bridge | budgeted assembly, two-phase receipts, proof-guided exact-message recovery | provider routing, normalized transcript, delivery and retries |
| OpenCode | local MCP server + compaction plugin | selection/verification tools, exact `ccr:` recovery, evidence-preserving compaction policy | session store, provider calls, compaction execution, code tools |

## Portable exact-recovery rule

A visible `ccr:<24-hex>` handle is a content address, not a search query.

1. Discovery tools decide which evidence may matter.
2. Entroly emits a `ccr:` handle when exact omitted content is recoverable.
3. Recovery accepts only that exact hash and returns the complete stored original.
4. A missing historical hash remains a miss; it never resolves to a newer source
   revision.
5. Recovery output must be protected from immediate re-compression by the host.

This separation prevents a relevance ranker from silently returning an empty or
partial result for content that is known to exist.

## Support boundaries

- No integration sends provider credentials to Entroly.
- No integration may claim provider savings without observing the provider-bound
  request and an explicit baseline.
- Host transcripts remain authoritative; Entroly stores only the exact recovery
  material and local receipts required by the enabled path.
- All adapters fail open to the host's original context when Entroly cannot safely
  produce a valid replacement.
