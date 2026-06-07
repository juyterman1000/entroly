# Entroly — Canonical Architecture

Entroly is an **auditable context control plane**. For any context it:

1. Takes any context.
2. Secures it.
3. Normalizes it.
4. Scores it mathematically.
5. Compresses / selects / distills it **safely**.
6. Proves the decision with a **receipt**.
7. Verifies hallucination risk.
8. Routes to the cheapest **safe** model (opt-in).
9. Preserves provider compliance.
10. Verifies the output.
11. Learns **privately** from outcomes.
12. Promotes only **proven** policy improvements.

---

## Invariants (non-negotiable)

- **Fail-closed — only stages 2 and 11.** The Security/Privacy/Compliance gate and the Provider Gateway compliance contract never degrade. Unsafe input is rejected or stripped; it is never forwarded.
- **Fail-open — everything else.** On uncertainty, low confidence, or internal error, Entroly passes the **exact original context through unchanged** and **emits a warning in the receipt**. Fail-open never means "silently compress anyway" and never means "break the request."
- **Provider compliance.** Preserve provider protocol, headers, tools, parameters, ordering, and cache semantics. Transform **only** injected context and output. **Never mutate model/params/tools** *unless the user explicitly enables transparent routing* — and then the swap is recorded in the receipt.
- **Receipt = the trace.** A receipt id is opened at the security pass, threaded through every stage, and finalized at stage 8. Tiered: lightweight by default, full receipt on high-risk or on demand.
- **Verification is risk-gated.** Stage 6 runs **cheap pre-flight risk** (coverage gap, entropy signal) to set the token budget and verification depth. Stage 13 runs **post-flight** grounding cheaply on every answer; the expensive verifiers (STAVE/TRIAD/CAVE/PROVE) run **only when flagged**.
- **FORGE repair is explicit.** If repair issues another model call it must be explicit, budget-controlled, provider-compliant, and **logged in the receipt** — never a hidden extra LLM call.
- **Learning is async, gated, and canaried.** It runs off the request path. A candidate policy is promoted only if it reduces cost/latency, does not reduce quality, does not increase hallucination risk, violates no compliance/privacy/provider rule, and passes golden tests — then a **canary with auto-rollback** guards live regressions before atomic publish.
- **CogOps is strictly additive.** Beliefs/vault/memory enrich selection, verification, receipts, and routing but are **never a hard dependency**; they degrade gracefully when stale or empty.
- **Local proof, zero provider calls.** `entroly simulate` and `entroly perf` produce savings and quality estimates without any API cost.

---

## Flow

```mermaid
flowchart TD
  U["Agents / Apps / Users"]

  subgraph REQ["REQUEST PATH — synchronous, latency-critical"]
    direction TB
    U --> ENTRY["1 · ENTRY (thin wrappers)<br/>SDK · CLI · Proxy · MCP · npm · integrations<br/>⟂ FAIL-OPEN → exact passthrough on wrapper error"]
    ENTRY --> SEC{"2 · SECURITY / PRIVACY / COMPLIANCE<br/>🔒 FAIL-CLOSED<br/>auth · path-safety · injection · PII/secrets · protocol"}
    SEC -->|"unsafe"| REJECT["reject / strip — never forward"]
    SEC -->|"pass → open TRACE = receipt id"| ROUTER

    ROUTER{"EPISTEMIC ROUTER<br/>intent · belief coverage · freshness · risk<br/>→ picks 1 of 5 flows"}
    ROUTER -->|"fresh + low-risk<br/>⚡ FAST PATH (most requests)"| FAST["cached/known answer<br/>or minimal selection<br/>(skips heavy stages)"]
    ROUTER -->|"stale / novel / high-risk<br/>FULL PATH"| NORM

    NORM["3-4 · NORMALIZE + INDEX/FINGERPRINT<br/>⟂ FAIL-OPEN<br/>chunk ids · token+byte offsets · repo map · depgraph · scaffold"]
    NORM --> ENGINE["5 · MATH CONTEXT ENGINE — Rust SSOT<br/>⟂ FAIL-OPEN<br/>BM25 · entropy · knapsack · submodular · dedup · depgraph · qccr · channel · cache-stable"]
    ENGINE --> PRE{"6 · POLICY + PRE-FLIGHT RISK (cheap)<br/>coverage-gap · entropy signal<br/>→ sets budget + verification depth"}
    PRE --> COMPRESS["7 · COMPRESS / SELECT / DISTILL<br/>⟂ FAIL-OPEN = exact passthrough (never unsafe compression)<br/>preserve deps · citations · tool structure · JSON shape · high-risk evidence"]
    FAST --> COMPRESS

    COMPRESS --> RECEIPT["8 · CONTEXT RECEIPT  (= the trace)<br/>selected/omitted + why · fingerprints · ratio · warnings · repro-hash<br/>↕ tiered: light by default · full on high-risk / on-demand"]
    RECEIPT -.->|"simulate / perf — NO provider call"| SIM["LOCAL PROOF<br/>savings + quality estimate · zero API cost"]
    RECEIPT --> ROUTE{"10 · ROUTING / CACHE / COST<br/>🟢 OPT-IN model routing — within user pool · transparent · in receipt<br/>fail-closed escalation · cache-align · cost cortex · value tracker"}
    ROUTE --> GW["11 · PROVIDER / LOCAL GATEWAY<br/>🔒 preserve protocol · headers · tools · params · cache semantics<br/>transform context/output ONLY · routing opt-in+transparent+in-receipt<br/>OpenAI · Anthropic · Gemini · local"]
    GW --> RESP["12 · MODEL RESPONSE"]
    RESP --> VERIFY{"13 · POST-FLIGHT VERIFICATION (risk-gated)<br/>cheap ALWAYS: WITNESS/EICV grounding<br/>deep ONLY if flagged: STAVE · TRIAD · CAVE · PROVE<br/>FORGE repair: explicit · budgeted · logged in receipt · NO hidden LLM call"}
    VERIFY -->|"pass"| OUT
    VERIFY -->|"flag → suppress / repair"| OUT
    OUT["14 · OUTPUT CONTROL → RESULT<br/>answer · tool result · code edit · receipt link · savings/risk summary"]
  end

  OUT --> STORE[("PRIVATE PROJECT STORAGE — .entroly/, local-only<br/>SQLite DB / JSONL fallback · receipts · feedback · ravs events · tuning_config<br/>retention + compaction · optional encryption/keyref · NO raw global sharing")]

  subgraph LEARN["LEARNING PLANE — async, OFF the request path"]
    direction TB
    DAEMON["daemon · PRISM-5D · autotune · reward crystallizer · evolution · skills"]
    GATE{"PROMOTION GATE<br/>cost↓ · quality↛ · halluc↛ · compliance✓ · golden tests"}
    CANARY{"CANARY in prod<br/>auto-rollback on live regression"}
    POLICY[["ACTIVE RUNTIME POLICY<br/>budgets · PRISM weights · thresholds · retention"]]
    DAEMON --> GATE --> CANARY --> POLICY
  end

  STORE -.-> DAEMON
  POLICY -. "atomic publish" .-> ENGINE
  POLICY -. .-> PRE
  POLICY -. .-> ROUTE

  subgraph COG["COGOPS / MEMORY — strictly ADDITIVE, never a hard dependency"]
    direction TB
    VAULT["beliefs · vault · flow orchestrator · verification engine<br/>change pipeline · long-term + Kanerva memory · blast radius"]
  end

  STORE -.-> VAULT
  VAULT -. "enrich · graceful if stale/empty" .-> ENGINE
  VAULT -. .-> PRE
  VAULT -. .-> VERIFY
  VAULT -. .-> RECEIPT
```

**Legend:** 🔒 **FAIL-CLOSED** (stages 2 & 11 only — never degrade) · ⟂ **FAIL-OPEN** (pass exact original context + warn in receipt; never silently compress; never break the request) · 🟢 **OPT-IN** · ⚡ **fast path** (most requests short-circuit) · solid = synchronous request path · **dashed = asynchronous** (learning loop + additive CogOps enrichment + local-proof branch).

---

## Storage rules

| Class | Destination |
|-------|-------------|
| Raw private data | project-local `.entroly/` **only** (SQLite DB, JSONL fallback) |
| Sanitized aggregates | optional user-global `.entroly/` (**opt-in**) |
| Human instructions | `AGENTS.md` / `CLAUDE.md` **only** |

**Maintenance:** retention policy, compaction, optional encryption / key-ref.

**Never store** raw prompts, tool outputs, documents, provider responses, or learned weights inside `AGENTS.md` / `CLAUDE.md`. **No raw global sharing.**

---

*This document is the canonical reference for Entroly's request and learning flow. The implementation is migrating to a Rust single-source-of-truth core (`entroly-*` crates) with Python (PyO3), npm (wasm-bindgen), and the CLI as thin wrappers; the stages above describe behavior, not language boundaries.*
