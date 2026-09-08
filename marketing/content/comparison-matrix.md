# Context Management: Feature Comparison

How does Entroly compare to other approaches for managing AI agent context?
This matrix compares **approaches**, not specific competitor products. All
Entroly claims are verifiable with `entroly verify-claims`.

---

## Approach Comparison

| Capability | Raw Context (no tool) | Manual File Selection | Generic Compressor | RAG / Vector Search | **Entroly** |
|---|---|---|---|---|---|
| **Context size reduction** | ✗ | Manual | ✓ ratio only | ✓ retrieval | ✓ budget-aware selection |
| **Exact recovery of omitted content** | N/A | N/A | ✗ lossy | ✗ different query = different chunks | ✓ content-addressed, byte-exact |
| **Audit trail (receipts)** | ✗ | ✗ | ✗ | ✗ | ✓ Context Receipts |
| **Knows what was left out and why** | N/A | Only you know | ✗ | ✗ | ✓ receipt includes omissions + risk |
| **Answer verification** | ✗ | ✗ | ✗ | ✗ | ✓ WITNESS (local, no API) |
| **Cache-aware prefix stability** | ✗ | ✗ | Often breaks cache | ✗ | ✓ stable prefix alignment |
| **Code structure awareness** | ✗ | Partial | ✗ text only | ✗ text only | ✓ AST, graphs, architecture |
| **Works with existing tools** | ✓ native | ✓ manual | Varies | Framework-dependent | ✓ MCP, proxy, SDK, CLI |
| **Setup effort** | None | Per-task manual | Varies | Significant | `pip install entroly && entroly go` |
| **Privacy / local-first** | Depends on provider | ✓ | Varies | Often needs embeddings API | ✓ local analysis, no outbound |

---

## When each approach wins

### Raw context (no tool)
**Best for:** Small codebases and short prompts that fit the context window
comfortably. If your entire repo is under a few thousand tokens, any context
tool adds overhead for no benefit. Entroly detects this and passes through
unchanged.

### Manual file selection
**Best for:** Developers who know exactly which 2–3 files are relevant and
prefer manual control. Good for one-off tasks, but doesn't scale to multi-turn
agentic sessions or large codebases.

### Generic compressor
**Best for:** Reducing token count on already-selected content. Useful when
you've already picked the right context and just need it smaller. Limitation:
no recovery, no receipts, no verification, and can break provider cache
prefixes.

### RAG / Vector search
**Best for:** Finding semantically similar content across large document
collections. Limitation: similarity ≠ evidence. RAG answers "what text looks
similar?" but not "what code is structurally related?" or "is the retrieved
chunk still current?"

### Entroly
**Best for:** Large and medium repositories where an AI agent needs
repository-aware evidence selection, exact recovery, audit trails, and
verification. Especially valuable for:
- Multi-turn agentic sessions where context compounds
- Teams that need to audit what an agent was told
- Workflows where one omitted line can change the answer (code, contracts, logs)

---

## Entroly's honest limitations

| Limitation | Detail |
|---|---|
| **Not magic** | Compression trades size for risk. SQuAD benchmark shows 90% retention at 43.8% savings — accuracy dropped from 80% → 72%. |
| **Pass-through on small inputs** | If your context already fits, Entroly does nothing and tells you. |
| **Proxy still sends to provider** | Entroly reduces what's sent, but selected content still goes to your configured AI provider. |
| **WITNESS is not perfect** | Local hallucination detection has measured false positives and false negatives. |
| **Workload-dependent** | Savings vary. Run `entroly simulate` on YOUR repo to see YOUR numbers. |

---

## Verify for yourself

```bash
pip install -U entroly
cd /path/to/your/project
entroly verify-claims    # bounded checks, no API key
entroly simulate         # your repo's context profile
```

*Apache-2.0 · [github.com/juyterman1000/entroly](https://github.com/juyterman1000/entroly)*
