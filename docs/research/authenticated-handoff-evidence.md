# Authenticated handoffs and current verification evidence

Status: research slice on branch `research/authenticated-handoff`. The
measurement plan was preregistered and pushed before any implementation
existed: `benchmarks/HANDOFF_EVIDENCE_PREREGISTRATION.md`, commit `1e4af8b9`.
The results are in section 7.

## 1. Product brief

**Buyer.** Platform engineering teams that run long-lived coding agents across
providers and sessions: Claude Code, Codex, Gemini, OpenClaw workers and CI
bots.

**Why agents switch midway.** Several things cause a switch: budget or rate
limits, a context window that fills up, a provider outage, a specialist
reviewer, a laptop-to-cloud move, or a session that ended.

**The cost of a wrong next action.** A successor that reuses "tests passed"
after a dependency, configuration or data change ships a regression on
evidence that no longer applies. A successor that reruns everything pays the
full verification cost again, and each agent in a chain pays it.

**Today's workaround.** Paste the previous agent's summary and either trust
it or rerun everything. Native continuations carry the transcript and
authenticate the human account, not the successor agent, and they do not
recheck carried results (section 3).

**The smallest trial on a team's own repository.**

1. `entroly-work evidence run -- <test command>` at handoff time.
2. `entroly-work handoff --evidence <id>`.
3. The host signs an assertion for the successor with
   `entroly-work issue-assertion`, using a key that never enters the agent's
   sandbox.
4. The successor runs `entroly-work accept`, which returns reuse, revalidate
   or unknown for each handed-over result, and the authority epoch.

No network call is made and nothing is uploaded.

## 2. What existed on `main` (`7e8d2bc5`)

| Concern | Before this branch | Where |
|---|---|---|
| Successor identity in a handoff | A free label. `--to-agent codex` is a string, and any process that can reach the state directory can act as it | `work_graph_cli.py` `handoff`/`handoffs`; `work_graph_mcp.py` `work_handoff`; Rust `HandoffReceipt.to_agent` (`entroly-engine/src/work_graph.rs`) |
| Handoff receipt integrity | Unkeyed SHA-256 commitment: tamper-evident against accidents, not against anyone who can recompute it | `work_graph.rs` `handoff_receipt` / `verify_handoff_receipt` |
| "Is the handoff still current?" | Equality of the whole source state (branch, HEAD, changed-file digests). Any commit or edit makes it stale. Environment, interpreter and installed-dependency changes are invisible | `work_graph_review.py` `_source_state`; `work_graph_inbox.py` `pending_handoffs` |
| Verification results | Review `--check NAME EXIT_CODE` values are caller statements, documented as such. `VerificationRecord.freshness` is `Stale` on any repository-commitment change, and no caller binds it to a command's inputs | `docs/agent-continuity.md`; `entroly-engine/src/engine_contracts.rs` `VerificationRecord::freshness` |
| Recovery acknowledgement | A digest-bound gate, not bound to any identity. Two concurrent acknowledgements both succeed | `work_graph_recovery_ack.py` `acknowledge` |
| Governance identity | Symmetric `egov1` token. See section 6 for three ways a label is taken as authority | `governance/identity.py`, `governance/domain.py`, `governance/policy.py` |
| Signed artefacts with pinned keys | Proof bundles report `verified` only for a pinned key | `proof_bundle.py` |

The OpenClaw bridge (`openclaw_bridge.py`) assembles context and signs
receipts. It has no handoff path, so the slice starts on the `entroly-work`
CLI, the smallest supported entry point. The MCP `work_handoff` tool is fenced,
but it cannot accept handoffs yet. The `research_harness.py` named in the
brief was not present in this environment, and the session had no connected
local folders.

## 3. Competitors and prior work (summary)

The full map, with every claim marked as code-read, run, source or vendor
description, is kept outside Git with the raw research. These are its
conclusions:

- **Native continuation** in OpenAI Responses and Codex resume/fork/apply,
  Claude Code resume/teleport, the Agent SDK, Gemini CLI, Jules, Copilot and
  Cursor authenticates the human account and carries a transcript or branch.
  None authenticates the successor agent or rechecks a carried result.
  Claude Code documents that resuming one session in two terminals
  interleaves both writers (vendor description).
- **Host controls.** The OpenAI article on persistent OpenClaw agents and
  OpenClaw Enterprise cover sandboxing, egress, credential brokering and a
  service identity per agent, but not verification at the handoff boundary.
  OpenClaw's session controller fences turns inside one gateway. A proposed,
  unimplemented OpenClaw Enterprise RFC adds lease-bound handoff for
  replacing the same agent.
- **MemTrace v3** (arXiv 2610.04838) gates memory reuse on alignment with the
  repository. Its released code marks every test result stale on any
  workspace revision, which is safe but allows no reuse after an unrelated
  edit. The paper does not model dependency, configuration or runtime
  inputs, or identity.
- **Handoff Debt v2** (arXiv 2606.02875) measures how much handed-over context
  reduces successor effort (20–59% fewer events). It does not measure stale
  evidence accepted or redundant reruns.
- **Dependency-aware caches** are the classical control: verifying traces
  (*Build Systems à la Carte*), Ekstazi, pytest-testmon, Bazel, Pants, Nx and
  Turborepo. The strongest one runnable here is pytest-testmon 2.3.0. Its
  source tracks only `.py` files that coverage saw execute. It records
  dependency versions as major.minor, dropping the patch, and tracks
  environment variables only through an opt-in expression that fails open.
  The research run and the preregistered run here both reproduce these
  behaviours.

Not measured, and why: MemTrace and the Handoff Debt harness need a model
endpoint and Docker or vLLM. Bazel, Pants and Buck2 could not be fetched
within this environment's network limits. Nx, Turborepo, Gradle and Ekstazi
need Node or the JVM. No provider credentials or host controls were
available.

## 4. Hypothesis

H1 to H3 are in the preregistration, section 2. In short:

- **H1 (safety).** Binding a result to what the verification actually read
  gives zero stale acceptances where the baselines accept some.
- **H2 (efficiency).** At least 20% fewer redundant verification operations
  than testmon. This was expected to tie or lose.
- **H3 (authority).** All 20 fail-closed authority cases pass.

## 5. Specification

### 5.1 Authorization predicate

Let `A` be an assertion, `T` the pinned trust anchors, `O` the inbox offer, `L`
the epoch log of workstream `w`, and `now` the verifier's clock. Acceptance is
allowed iff every one of the following holds. Each is checked in
`handoff_authority.accept_handoff`:

```
shape(A)                       exactly {schema, claims, kid, signature}; claims exactly
                               {iss, aud, sub, scope, workspace, task, handoff, jti, iat, nbf, exp}
pinned(A.kid, A.iss, T)        the key named by kid is pinned for iss
Ed25519.verify(A.sig, D || canonical(A.claims))   D = "entroly.handoff-assertion.v1\0"
A.aud = T.audience
A.exp - A.iat <= T.max_lifetime
A.nbf - skew <= now <= A.exp + skew
"work:accept" in A.scope
A.workspace = repo_id = O.repo_id
A.task = O.workstream
A.handoff = sha256(canonical(O))                  O commits to the receipt, the workspace
                                                  commitment and every evidence digest
A.sub = O.to_agent
no pending recovery acknowledgement
L verifies as a contiguous hash chain
sha256(A.jti) not in L, except as the idempotent retry of the same offer at the head
O.base_epoch = |L|
```

The transition is `L := L ++ [epoch |L|+1]`. It is published with `os.link`
of a fully written private file, which is the single linearization point. The
winner is whoever links `epoch-(|L|+1)`. A crash before the link changes
nothing. A crash after it is recovered by repeating the same assertion.

**Fenced writes.** These are `handoff`, `evidence run --workstream` and MCP
`work_handoff`. On a held workstream they require an assertion with
`work:write`, the same workspace and task, `sha256(jti)` and `sub` equal to the
head epoch's, and `now <= exp + skew`. The MCP tool takes no assertion, so on a
held workstream it always refuses.

### 5.2 Dependency closure

For one traced run of command `c`, closure(c) has these parts:

| Part | Contents | Bound as |
|---|---|---|
| R | Files opened for reading by any traced process, plus the files behind every module loaded at exit (this covers extension modules). A cached `.pyc` is mapped to its source | Content SHA-256 |
| P | Paths probed with `os.stat`/`os.lstat`. This is how pytest looks for `conftest.py` and configuration files, and probes of absent files catch later creation | absent / file / dir |
| L_full | Directories listed by any code other than the two consumers below | Sorted name list |
| L_import | Directories listed by the import system's directory cache | Names whose module stem was ever imported, plus `.pth`, `*.dist-info`, `*.egg-info` and `.egg-link` entries |
| L_meta | Directories listed by `importlib.metadata` | `*.dist-info` and `*.egg-info` names |
| E | Environment variables read through `os.environ` that the process had not set itself | Salted value digest, or `absent` |
| X | `argv[0]` resolved on `PATH` | Invoked path, real path and content |
| Root | The project root | Its real path |

**Exclusions.** Tool caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`,
`.ruff_cache`, `.hypothesis`, `.coverage*`, `.testmondata*`) are excluded.
Outside the project root, so are the system temp directory, Entroly's state,
`/proc`, `/sys` and `/dev`. A path the run wrote and also read makes the
record unknown.

**Trace completeness.** The trace is complete iff three conditions hold:

- the command's own process was traced;
- every traced process wrote its exit record;
- no process started more children than were traced.

### 5.3 Applicability predicate

`assess(record, now)`:

- **unknown** if the record fails its own digests, its digest differs from the
  one the accepted offer bound, the trace is incomplete, an input cannot be
  read, or the Work Graph's `VerificationRecord` (with
  `verified_repository_commitment` = digest of closure) is not `current`.
- **revalidate** if any closure entry's digest differs now. The changed
  entries are listed.
- **reuse** otherwise.

`unknown` is executed as revalidate. A failed result that still applies is
reused as a failure.

Provenance is reported separately from applicability:

- who recorded the result (a label, never authenticated);
- whether Entroly observed the exit code itself;
- the authority epoch it was recorded under;
- whether the record digest was bound by the host-signed offer.

### 5.4 Soundness assumptions and known false stops

The decision is sound for reuse under these assumptions:

1. The outcome is a deterministic function of the closure. Clock time,
   randomness, network responses and hardware are not bound.
2. Native code is opaque. File reads and `getenv` calls from C extensions,
   and `stat` calls by the import system itself, are not observed. A
   namespace package gaining `__init__.py` is caught only through the import
   listing.
3. Excluded tool caches do not change outcomes. For example, a stored
   Hypothesis example, a `--lf` selection or a `.pyc` with the same mtime and
   size as its source is not an input.
4. Listings and `stat` calls relative to a directory descriptor are not
   resolved. `shutil.rmtree` and `os.fwalk` make them.
5. The workspace is honest. Records and epoch logs are unkeyed digests, not
   signatures.

The trace is marked incomplete, and the result is never reused, in these cases:

- a process that is not traced, including a launcher that `exec`s Python and
  keeps its pid;
- a lost trace event;
- an environment that was enumerated;
- an input modified or replaced by the run;
- an input whose size or mtime changed during the run.

Known conservative behaviour (a correct but redundant rerun):

- New names in a directory pytest scanned (case C4).
- Presentation variables pytest reads (`COLUMNS`, `NO_COLOR`).
- Any package install in the environment.
- Every `PYTHON*` start-up variable.

## 6. Security and privacy review notes

- **The caller chooses the trust file.** A local process can generate a key,
  write its own trust file and accept a real offer as the named successor.
  Entroly cannot prevent this from inside the workspace. It records the trust
  file's digest and key id in the epoch. It labels the authority `advisory`
  unless the trust file and its directory are beyond the current user's reach
  (as with a read-only host mount; never as root). The host must check epoch
  key ids against its own keys. Acceptance binds to one state directory, so
  the same assertion can be accepted in another one; the host must record
  each attempt id it consumes.
- **The assertion is a bearer capability** for the attempt while it is valid.
  It is written 0600 and not stored in the state directory. Only its digest
  and the digest of its `jti` are stored.
- **The evidence record stores digests only.** It holds no file contents and
  no command output. Environment values are salted hashes. Paths are kept in
  the private (0700) state directory. Receipts and epoch records carry
  decisions, reasons, counts and digests.
- **Label mistaken for authority, found on `main` and not changed here.**
  - `AgentIdentity.is_verified` returns `bool(identity_token)`, so any
    non-empty string counts as verified (`governance/domain.py`).
  - With no `ENTROLY_IDENTITY_KEY`, `resolve_identity` accepts a forged
    identity carrying `admin` scope and reports `verified: true`.
  - With a key configured, an identity with no token keeps its self-claimed
    scopes.
  - Policies are chosen by the self-declared `agent_type`
    (`governance/policy.py` `find_matching_policy`).

  This was reproduced against 1.0.90. The slice does not build on governance
  identity for exactly this reason. The fix is a separate, compatibility-
  sensitive change: verify-or-downgrade whenever any claim beyond anonymous
  read is made, and report `verified` only after a key check.

## 7. Results

See `benchmarks/results/handoff_evidence.json` (challenge suite) and
`benchmarks/results/handoff_evidence_replay.json` (held-out click replay).
Each file records the exact head it ran on.

## 8. Independent review

Two reviewers with no part in writing the code reviewed `554bb380`, one for
security and one for production. Their reproductions changed the code before
the final runs:

| Finding | Severity | Change |
|---|---|---|
| A local process can pin its own key | high | `advisory` label, trust digest and key id in the epoch, documented host check |
| Evidence records are unsigned | high | `record_is_signed: false` in provenance; a record with no traced reads is `unknown`; no claim of host inspection |
| A shell that `exec`s Python looked fully traced | high | The root's own argv must match the command (direct or shebang), else `command-started-by-untraced-program` |
| One re-entrancy flag for all threads dropped reads | high | Per-thread guard; dropped events mark the trace incomplete |
| A missing audit hook still wrote a complete trace | high | Exit handler registered only after the audit hook |
| `--timeout` hung on grandchildren | high/medium | Process group killed; output streamed to private files |
| Start-up inputs (`PYTHONPATH`, `.pth`) were unbound | medium | Every `PYTHON*` start-up variable and each `.pth` file's contents bound |
| Rename/remove after a read was not a write; inputs changed during the run | medium | Mutation audit events recorded as writes; first-read size and mtime compared after the run |
| `os.access`, and new `__init__.py` in import listings | medium | Traced as probes; `__init__` kept in listings |
| Tracing disabled `rmtree`'s symlink defence | medium | Wrappers added to `os.supports_*` |
| Future `iat` stretched the lifetime | low | `iat <= now + skew`; lifetime measured from `min(iat, nbf)` |
| Low-entropy environment values could be brute-forced from the record | low | Digests keyed by a state-directory secret |
| `evidence run` exited 0 on a failed command | medium | Exits with the command's code; `check` exits 0 only for reuse |
| Fence ignored issuer and key; a malformed record crashed `accept` | low | Both fixed |
| One winner per state directory; deleting the log unfences; only three writes fenced; argv stored verbatim | medium/low | Documented, not fixed |

The reviewers' reports are kept with the private evidence, outside Git.
