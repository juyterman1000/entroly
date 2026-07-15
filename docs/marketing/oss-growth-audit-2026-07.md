# Entroly open-source growth and trust audit

Snapshot: 2026-07-15. Owner: Entroly maintainers. Review quarterly.

This audit treats adoption as an engineering outcome: a developer must discover
Entroly, understand its boundary, reach a verified result quickly, and remain
confident that upgrades will not lose data or silently change behavior. It does
not recommend paid stars, mass outreach, keyword stuffing, or unverified claims.

## Executive diagnosis

Entroly has strong technical breadth and unusually concrete proof artifacts, but
its conversion system is weaker than its engineering system.

The supplied 14-day GitHub traffic snapshot shows 6,273 clones from 597 unique
cloners, compared with 1,144 page views from 159 unique visitors. The supplied
ClawHub snapshot shows six downloads for version 1.0.60. That mismatch suggests
significant machine, CI, or existing-user activity without equivalent human
discovery and marketplace activation. It does not prove that every clone is a
new user.

The highest-leverage correction is not a louder claim. It is a shorter,
evidence-first path:

1. One problem: agents lose or distort the evidence behind an answer.
2. One promise: Entroly records exactly what context was selected and omitted.
3. One no-key proof: `entroly verify-claims && entroly simulate`.
4. One client-specific installation choice.
5. One inspectable receipt before any benchmark claim.

## Repository scorecard

| Area | Evidence | Assessment | Priority |
|---|---|---|---|
| Product differentiation | Context selection, recoverability, receipts, verification, proxy, MCP, OpenClaw | Distinctive but presented as too many products at once | P0 |
| README | Real proof videos, scoped claims, install paths, compatibility | Trustworthy but long; user choice arrives amid a large feature inventory | P0 |
| First success | Local no-key verification exists | Needed a dedicated short quickstart and copy/paste example | P0 |
| Documentation | Large docs set and product-surface map | Core API, architecture, support, migration, and troubleshooting entry points were missing at root | P0 |
| Community | Discussions enabled; community health reported complete | Only two visible contributors in the snapshot; generic intake and unclear decision rights | P0 |
| Security | Secret scanning and push protection enabled | CodeQL and dependency review added in this branch; `pip-audit`, `cargo-audit`, and npm audit still need a clean baseline | P0 |
| Branch governance | Required checks with strict updates | Force pushes allowed; conversation resolution and admin enforcement disabled | P0 owner action |
| CI | Python 3.10-3.14, Rust, fallback, proxy, OpenClaw and release gates | Strong matrix; this branch adds trust-critical coverage, link and spelling gates | P1 |
| Dependency maintenance | Dependabot for root pip, core Rust and Actions | Nested Rust, npm, WASM and OpenClaw manifests were uncovered | P1 |
| Architecture | Python plus optional Rust and Node/WASM | Several modules exceed 4,500 lines, raising review and ownership risk | P1 |
| Releases | Automated multi-platform publication; rapid cadence | Frequent micro-releases can create upgrade fatigue without a predictable train | P1 |
| Website | Dedicated landing page and dashboard | Repository homepage points to the dashboard instead of the landing page | P0 quick win |
| SEO | Relevant description and 19 topics | Missing the canonical `context-engineering` and `openclaw` discovery terms | P0 quick win |
| Branding | Consistent wordmark and dark visual system | No upload-ready social card existed before this work | P1 |
| Benchmarks | Raw artifacts, caveats and local repro commands | Results are fragmented; a versioned scorecard and independent reproduction path are needed | P0 |

## Comparable-project snapshot

GitHub counts change continuously. Stars, forks, contributors, and releases are
rounded snapshots collected from public GitHub repository data on 2026-07-15.
Release counts cover the preceding 90 days and are capped at `100+`. Qualitative
notes describe visible public repository surfaces, not private community health
or product quality.

| Project | Stars | Forks | Contributors | Releases / 90d | Public-repository comparison |
|---|---:|---:|---:|---:|---|
| `headroomlabs-ai/headroom` | 59,216 | 4,391 | ~202 | 100+ | Excellent 60-second onboarding, demos, compatibility and Discord; broad proxy/compression platform; Entroly should win on inspectable receipts and recovery evidence, not negative comparison. |
| `yvgude/lean-ctx` | 3,247 | 300 | 45 | 100+ | Strong name, visual demos, scenarios and short install; polished project surfaces; Entroly needs an equally short first-success path. |
| `fkiene/llmtrim` | 165 | 8 | 2 | 35 | Minimal problem narrative and before/after example; narrow API-proxy scope; Entroly has more capability but a heavier decision surface. |
| `microsoft/LLMLingua` | 6,435 | 399 | 17 | 0 | Research credibility, papers and reproducible examples; mature API story but slower release cadence; Entroly needs a citable technical report. |
| `NVIDIA/kvpress` | 1,134 | 159 | 28 | 1 | Focused KV-cache compression, strong institutional brand and benchmark framing; narrower runtime target than Entroly. |
| `mit-han-lab/streaming-llm` | 7,244 | 399 | 5 | 0 | Memorable research concept, paper and simple demo; not a general agent control plane. |
| `mem0ai/mem0` | 60,865 | 7,095 | 378 | 55 | Banner, managed-versus-OSS disclosure, benchmark table, SDKs, docs and community; Entroly needs similarly clear deployment boundaries. |
| `letta-ai/letta` | 23,803 | 2,520 | 153 | 1 | Strong agent-memory category ownership, API/docs, examples and community; Entroly's differentiator is request-level evidence accountability. |
| `getzep/graphiti` | 28,734 | 2,902 | 44 | 3 | Clear temporal knowledge-graph thesis and integrations; high-quality brand and examples; operationally heavier than Entroly's local-first path. |
| `MemTensor/MemOS` | 10,213 | 930 | 93 | 8 | Research-led memory operating-system positioning, papers and broad examples; Entroly should avoid overlapping OS language. |
| `supermemoryai/supermemory` | 28,385 | 2,471 | 97 | 12 | Strong consumer/developer brand, hosted path, SDKs and social surface; Entroly offers stronger local auditability. |
| `memodb-io/Acontext` | 3,579 | 325 | 10 | 0 | Focused agent context/data layer and concise onboarding; early release maturity. |
| `upstash/context7` | 59,123 | 2,792 | 128 | 28 | Category-defining problem statement, one-command setup and many client integrations; excellent search intent and distribution. |
| `BerriAI/litellm` | 53,635 | 9,777 | ~1,541 | 96 | Provider breadth, docs, enterprise path and huge contributor network; Entroly should integrate rather than imitate its gateway breadth. |
| `Portkey-AI/gateway` | 12,430 | 1,206 | 123 | 0 | Focused AI gateway, observability, enterprise docs and integrations; Entroly differentiates through context provenance. |
| `Helicone/helicone` | 5,948 | 627 | 101 | 0 | Strong observability dashboard, hosted onboarding and content funnel; Entroly's receipt UI can make invisible context decisions inspectable. |
| `langfuse/langfuse` | 31,177 | 3,286 | 191 | 62 | Excellent observability category, SDKs, docs, integrations and community; sets the bar for stable UX and release notes. |
| `openlit/openlit` | 2,603 | 324 | 93 | 27 | OpenTelemetry-native positioning, examples and integration breadth; Entroly can export context events into this ecosystem. |
| `run-llama/llama_index` | 50,859 | 7,754 | ~1,963 | 3 | Extensive framework, docs, templates, courses and community; powerful but complex. Entroly should remain an interoperable control layer. |
| `langchain-ai/langchain` | 141,809 | 23,559 | ~3,698 | 95 | Category scale, comprehensive APIs, tutorials, integrations and social reach; not a direct compressor, but a critical adoption channel. |

### What the leaders consistently do

- Lead with one painful job rather than a catalogue of subsystems.
- Put a 30-60 second install and a visible result above the architecture tour.
- Show a real workflow, compatibility matrix, and explicit “use this when” and
  “do not use this when” guidance.
- Own a search phrase naturally through docs, examples, package metadata and
  external technical references.
- Convert users into contributors through bounded issues, public roadmaps,
  responsive review, and visible recognition.

## Gap analysis

### Discovery

Entroly's strongest search intent is “auditable AI context,” “context receipts,”
“recoverable context compression,” and “OpenClaw context engine.” The repository
currently spreads attention across context OS, memory, gateways, hallucination
guarding, compression and routing. Standardize on “auditable context control for
AI agents” and retain the narrower terms in the relevant guides.

### Activation

The no-key verifier is a competitive advantage. Every public surface should
route to it before asking the user to configure a provider. Measure median
time-to-first-success from a clean environment; the target is under five
minutes with a 90% completion rate.

### Trust

Every number should link to the exact artifact, command, dependency version,
scope and failure criteria. Competitive results should invite side-by-side
reproduction and never imply universal superiority. If a result cannot be
regenerated from a release tag, remove it from the top fold.

### Sustainability

Two visible contributors and several 4,500+ line modules create concentration
risk. Establish component owners, extract policy modules behind compatibility
tests, publish monthly good-first-issue batches, and report review latency.

### Release confidence

Move toward a predictable weekly or biweekly feature train with urgent patch
releases reserved for data loss, security, install failure and provider
breakage. Publish one human-readable upgrade summary per train.

## Changes implemented in this audit branch

- Five-minute quickstart, public API reference, architecture, examples,
  migration, FAQ, troubleshooting, support and best-practice guides.
- Current support, security, governance and maintainer policies.
- Structured bug, feature and evidence-report forms plus a trust-oriented pull
  request template and expanded code ownership.
- Offline internal-link validation, spelling enforcement and regression tests.
- CodeQL, dependency review and a measured trust-critical branch-coverage gate.
- Dependabot coverage for nested Rust, npm, WASM and OpenClaw manifests.
- Corrected package claim language and website metadata.
- Editable 1280x640 social-preview source and rendered PNG.
- Repository homepage now targets the landing page; description and topics now
  use the canonical context-control, context-engineering and OpenClaw language.
- Added triage, benchmark, performance, security, reproduction and component
  labels used by the structured intake forms.

## Owner actions that should not be hidden in a code change

1. Upload `docs/assets/social-preview.png` in repository settings.
2. Require resolved review conversations, disable force pushes, enforce checks
   for administrators, and consider signed commits on the protected branch.
3. Enable Dependabot security updates after confirming triage ownership.
4. Publish a private security contact that is not tied to one maintainer's
   personal mailbox.

## Metrics and monthly review

| Funnel | KPI | Baseline | 90-day target | Source |
|---|---|---:|---:|---|
| Discover | Unique GitHub visitors / 14d | 159 | 400 | GitHub Traffic |
| Discover | Search-referred unique visitors / 14d | 15 known Google/Bing | 75 | GitHub Traffic |
| Evaluate | README-to-quickstart click rate | Not instrumented | Establish baseline, then +30% | Privacy-preserving site analytics |
| Activate | Clean-install success rate | Not instrumented | >=90% | Opt-in CI/tutorial cohort |
| Activate | Median time-to-first verified receipt | Not instrumented | <5 minutes | Usability sessions |
| Adopt | PyPI/npm/ClawHub downloads | ClawHub: 6 supplied snapshot | +20% month over month for 3 months | Registry statistics |
| Retain | Repeat users / 30d | Not instrumented | Establish ethical, opt-in baseline | Aggregated opt-in telemetry only |
| Community | New contributors / month | 2 visible all-time contributors | 4 | GitHub Insights |
| Community | Median first response to issue | Not measured | <2 business days | GitHub API |
| Community | Median PR time to decision | Not measured | <5 business days | GitHub API |
| Quality | Escaped P0/P1 regressions | Not centrally reported | 0 P0; downward P1 trend | Incident log |
| Quality | Coverage on trust-critical modules | Not reported | Baseline then >=80% branch coverage | CI |
| Release | Failed or rolled-back releases | Not reported | 0 | Release workflow |
| Docs | Broken local links / default branch | No gate | 0 | Documentation quality workflow |
| Sustainability | Files >2,000 lines without an owner/refactor plan | Multiple | 0 unowned | Architecture review |

Do not collect file contents, prompts, repository names, provider keys or user
identifiers to measure growth. Prefer registry, GitHub and aggregate opt-in
events.

## Twenty highest-ROI actions

| Rank | Action | Impact | Effort | Success signal |
|---:|---|---|---|---|
| 1 | Make the no-key verifier the single primary call to action | Very high | <1 hour | More quickstart completions |
| 2 | Point the repository homepage at the landing page | High | <15 min | Higher docs sessions |
| 3 | Enforce accurate support/security/governance policies | Very high | Done in branch | Fewer ambiguous reports |
| 4 | Ship structured issue and PR intake | High | Done in branch | Reproduction-complete issues |
| 5 | Require resolved conversations and disable main force pushes | Very high | <1 hour, owner | Protected-main policy passes audit |
| 6 | Publish one versioned benchmark scorecard with raw artifacts | Very high | 3-5 days | Independent reproduction |
| 7 | Add coverage and security audit gates | Very high | 2-4 days | Trends visible; zero critical findings |
| 8 | Standardize positioning on auditable context control | High | 1 day | Higher search impressions/clicks |
| 9 | Add `context-engineering` and `openclaw` topics | High | <30 min | Topic discovery traffic |
| 10 | Upload the prepared social preview | Medium-high | <15 min | Better external-link engagement |
| 11 | Measure clean-install success and time-to-first receipt | High | 2 days | >=90% and <5 min |
| 12 | Create LangChain/LlamaIndex/LiteLLM integration examples | High | 3-5 days | Example installs and referrals |
| 13 | Establish a predictable release train | High | 1 day + discipline | Fewer patch releases and regressions |
| 14 | Decompose `cli.py`, `proxy.py`, and `server.py` behind contract tests | High | Multi-week | Smaller owned modules, stable API |
| 15 | Add a searchable receipt/session viewer | High | 1-2 weeks | Receipt inspection and retention |
| 16 | Publish monthly good-first-issue batches with mentors | Medium-high | 2 hours/month | Four new contributors/month |
| 17 | Add docs link/spelling gates | Medium | Done in branch | Zero broken default-branch links |
| 18 | Expand nested dependency update coverage | Medium | Done in branch | All manifests receive updates |
| 19 | Publish two evidence-led technical assets each month | Medium-high | Ongoing | Qualified referrals and citations |
| 20 | Retire stale or unsupported public claims and pages quarterly | High | 1 day/quarter | Zero unverifiable prominent claims |

## Release gate for public claims

A prominent claim is releasable only when a tagged artifact records the input,
dependency and model versions, command, seed where applicable, raw output,
metric definition, limitations, and a failing threshold. A maintainer who did
not author the benchmark should reproduce it. “Better” must name the metric,
workload, version and confidence interval or statistical test.
