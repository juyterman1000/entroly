# Structural context assurance

`entroly.context-assurance.v1` records deterministic facts about a declared
evidence selection. It does not certify model answers or decision invariance.
Public Python Context Receipts carry the certificate under
`risk_summary.context_assurance`; the existing receipt v1 hash includes it.
Native and Python receipts use the same Python assurance audit. The separate
Rust/WASM Work Graph receipt envelope is unchanged.

For an original evidence universe C, retained units S and omitted units C minus
S, the implemented hard contract checks:

- retained units are unique members of the declared index and retain its exact
  text, source identity, byte ranges and byte digests;
- exact-byte commitments cover the complete index, including omissions beyond
  the bounded 20-entry receipt preview;
- every declared mandatory unit is retained;
- each declared carrier obligation retains at least its specified minimum;
- resolved dependencies of retained units are retained and unresolved declared
  dependencies block the contract;
- the joined selected text fits the budget under local `o200k_base` counting.

Budget assurance covers the specified double-newline rendering of selected
text. It excludes receipt metadata, provider wrappers, system prompts and tool
messages. Other provider tokenizers require separate accounting. When the local
encoder is unavailable, the character heuristic cannot earn budget assurance.
Install the declared `entroly[test]` or `entroly[benchmark]` extra when exact
local budget assurance is required; the base-install path remains fail closed.
Canonical token counting only loads a verified local tiktoken asset or an
already initialized encoder. Missing or corrupt assets remain unavailable;
installing the optional package alone does not authorize an asset download.
An explicit online setup step can provision it with
`python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"` before going
offline. Runtime counting preserves corrupt cache files and uses the documented
heuristic when no verified asset is available.

The index is the trust anchor supplied by the caller. Digest consistency is not
an authenticated observation of the current filesystem. Mutating an index
invalidates an old selection against it; historical snapshot recovery still
returns historical bytes. Source provenance, detected dependency completeness,
and task truth are not established merely by possessing hashes.

## Joint omission and diagnostic boundaries

`HardObligation("route", (a, b), minimum=1)` permits omitting a or b individually
while rejecting their joint omission. All obligations are reevaluated against
the final retained set. Unknown carriers fail. Soft scores cannot override a
failed hard check.

Existing RELATE joint omission checks provide additional constraint, conflict,
value and action-argument vetoes. Lexical and dimension coverage diagnostics
remain separate estimates. These heuristics do not detect all task constraints
or prove semantic preservation. Work is bounded at 64 chunks and 256,000 UTF-8
bytes; an omitted-set check beyond that limit is unavailable, never a pass.
No raw diagnostic source text is added to the certificate.

Verdicts are `rejected`, `expansion_required`, `uncertain`, or
`structurally_valid_risk_unmeasured`. Even the strongest implemented verdict
leaves `decision_risk.upper_risk_bound` null. Internal selection regret is named
`internal_selection_regret_upper_bound`; downstream decision-divergence regret
and task context regret remain separate, unmeasured quantities.

## Public paths

| Path | Assurance authority |
| --- | --- |
| Public Python Context Receipt selection | Complete declared index, retained byte identities, declared dependencies, joint diagnostics |
| `entroly.assure_context` | Same authority with explicit mandatory/carrier obligations and a project/session/agent binding; rejects a failed contract |
| Engine / SDK `optimize` | Shared schema and risk boundary; source-span census and joint obligations unavailable, so no semantic assurance verdict |
| MCP `optimize_context` | Diagnostic recomputed after sanitizing and compacting actual returned selection |
| Proxy selection | Same diagnostic recomputed for the rendered context block after filtering, hierarchical rendering, memories and cache alignment; full provider messages still require independent transport accounting |
| Verified efficiency / fixed-point preparation | Public receipt extension is included in existing context commits and signed audits; verifier and recovery policies remain separate |

These paths share an assurance vocabulary. Their selector algorithms remain
different. QCCR excerpts or memory beliefs cannot become exact source evidence
through an optimizer score. No automatic calibrated risk enforcement is added
to provider traffic, and no new MCP tool is introduced.

## Scoped selection and recovery

```python
from entroly import AssuranceScope, assure_context, recover_context_evidence

scope = AssuranceScope(project="repo:app", session="task:123", agent="agent:1")
result = assure_context(
    [("facts.txt", "The service retry budget is seven attempts.")],
    query="service retry budget",
    budget=200,
    scope=scope,
)
selected = result["selected_context"]
certificate = result["certificate"]
# Each omitted handle binds the complete committed snapshot and declared scope.
for chunk_id in result["recovery_map"]:
    recovered = recover_context_evidence(result["context_commit"], chunk_id, scope=scope)
```

The commit and recovery map contain source evidence and must remain local.
Recovery never opens a recorded source path. It verifies the whole supplied
commit and exact UTF-8 snapshot bytes, including unlisted omissions. Restart
recovery uses the same serialized commit. Project, session or agent mismatches
are rejected. Scope binding is not principal authentication: callers must trust
the supplied commit or authenticate it using the existing signed audit before
crossing a trust boundary. Existing recovery stores use private POSIX file
permissions; Windows inherits parent ACLs. Recovery rejects unsafe receipt IDs
and symlinked artifact files. Trusted parent directory and local-user access
remain assumptions; no encryption or sandbox boundary is claimed.

`require_decision_risk=True` fails closed because this version has no calibrated
downstream risk authority. Larger budgets can be retried by the caller; this
operation does not claim an adaptive stopping or optimality guarantee.

## Evidence and limits

Deterministic declarations are **implementation property-tested**: bounded
carrier/dependency oracles, input permutations, source mutations, missing
metadata, hard vetoes, restart recovery and scope isolation. No Lean proof or
production refinement proof is supplied. No downstream quality, savings,
provider-spend, risk-calibration, adaptive-budget or mixed-resolution advantage
follows from these tests. Existing WITNESS/RAVS calibration is not transferred
to context-induced decision divergence.

Rollback is a scoped code revert. Existing receipt and commit schema names are
preserved; no persistent index migration or selection algorithm change occurs.
