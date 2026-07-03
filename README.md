中文 • 日本語 • 한국어 • Português • Español • Deutsch • Français • Русский • हिन्दी • Türkçe

Entroly

Cut your Claude / OpenAI / Gemini bill 70–95% on AI coding.
Compress context, keep provider caches hot, and verify every answer with a $0 hallucination guard.

Drop-in for Cursor, Claude Code, Codex, Aider + 34 more and custom providers — 30s, no code changes.

Auditable context control plane · every answer gets a receipt: what was used, what was omitted, why, and the risks that remain · local-first · Rust + WASM · reversible · savings measured on real workloads

PyPI npm License Token savings Hallucination guard Rust + WASM

pip install entroly && cd /your/repo && entroly go

Get started · Proof · Integrations · What's inside · Architecture · For teams · Limitations

Live demo   Dashboard

What it does
Entroly is an auditable context control plane for AI agents. It decides what context to send, records what it left out, and produces a receipt you can inspect before trusting a hard multi-file answer.

Receipts - every selection run can explain selected chunks, omitted nearby evidence, dependency links, fingerprints, token ratio, and residual risks.
Select - ranks your repo or document set, then sends the answer-relevant context under a token budget.
Verify - WITNESS checks the model's answer against the evidence it was given and flags unsupported claims. $0, ~3 ms, no extra API call.
Route - sends easy, repeated tasks to a cheaper model and keeps the flagship for hard ones (opt-in, fail-closed).
Cache-align - keeps the injected prefix byte-stable so provider prefix caches can keep hitting where terms and API shape allow it.
Learn - improves which files it picks for your workflow from local feedback. No embeddings API, no training job.
Use it however you work: wrap your agent, run it as a proxy, plug it in as an MCP server, or import the library.

How it works (30 seconds)
your agent  ──►  Entroly (local)  ──►  LLM provider
                 │
                 ├─ rank the repo        (BM25 + entropy + dep-graph)
                 ├─ select under budget  (knapsack, reversible)
                 ├─ emit receipt         (included, omitted, risks)
                 ├─ cache-align prefix    (keep provider cache hot)
                 └─ verify the reply      (WITNESS hallucination guard)
Critical files go in full. Supporting files become signatures. Everything else becomes a reference you can expand on demand — so the model gets a broader view of your codebase in a smaller prompt. Nothing is lost: every compressed fragment is fully retrievable.

Get started (60 seconds)
pip install entroly        # or: npm i -g entroly  ·  brew install juyterman1000/entroly/entroly
1. One command — auto-detects your IDE, wraps your agent, opens the dashboard:

cd /your/repo && entroly go
2. Or wrap a specific agent:

entroly wrap claude     # Claude Code
entroly wrap cursor     # Cursor
entroly wrap codex      # Codex CLI
entroly wrap aider      # Aider
3. Or run the proxy — zero code changes, any language:

entroly proxy                                   # http://localhost:9377
ANTHROPIC_BASE_URL=http://localhost:9377     your-app
OPENAI_BASE_URL=http://localhost:9377/v1     your-app
4. Or measure it on your own repo first:

entroly demo            # before/after token + cost estimate
entroly simulate        # local no-LLM savings estimate
entroly perf            # local no-LLM savings + optimizer latency
entroly verify-claims   # runs the packaged self-test, writes a JSON report
Local-first: your code is indexed and selected on-device, never sent anywhere for analysis. Apache-2.0. No outbound analytics by default.

Context Receipts
Entroly gives every AI answer a context receipt: what was used, what was omitted, why, and what risks remain. This is built for hard multi-document work such as contracts, policies, addenda, code reviews, and audit evidence where "top-k chunks" is not enough.

entroly ingest ./docs
entroly select --query "Does this contract have a change-of-control clause?" --budget 8000
entroly receipt .entroly/receipts/cr_example.json
entroly explain --why-omitted chk_example --receipt .entroly/receipts/cr_example.json
The receipt JSON includes selected chunks, omitted relevant chunks, ranking reasons, dependency links, source fingerprints, token ratio, warnings, and a reproducibility hash. The Markdown report is designed for human review before a compressed context is trusted.

Implementation notes:

Rust core (entroly-core/src/context_receipts.rs) handles deterministic ingestion, BM25-style ranking, dependency scans, selection, and hashes when the native wheel is available.
Python control plane (entroly/context_receipts/) provides CLI wiring and a pure-Python fallback for source checkouts.
The semantic/vector scorer and reranker are explicit extension points; the local MVP ships with lexical scoring and dependency heuristics, not a legal-accuracy guarantee.
Examples:

Example receipt JSON
Example Markdown report
Limitations
Proof
Every number below is reproducible and backed by a committed JSON artifact you can audit — not a screenshot.

Token savings (this repo, entroly verify-claims, local, no API):

Budget	Token reduction
8K	99.1%
32K	96.7%
average across workloads	87.0%
Accuracy retention — does compression hurt answers? Measured with gpt-4o-mini; intervals are Wilson 95% CIs. Each row links its raw result file.

Benchmark	n	Budget	Baseline	With Entroly	Retention	Token savings
NeedleInAHaystack	20	2K	100%	100%	100%	99.5%
LongBench (HotpotQA)	50	2K	64%	66%	103%	85.3%
Berkeley Function Calling	50	500	100%	100%	100%	79.3%
SQuAD 2.0	50	100	80%	72%	90%	43.8%
GSM8K	20	50K	85%	85%	100%	pass-through*
*pass-through: context already fit the budget, so Entroly left it unchanged. Reproduce: python benchmarks/run_readme_benchmarks.py (needs OPENAI_API_KEY). Full table + MMLU/TruthfulQA in DETAILS.

Hallucination guard — HaluEval-QA, standard protocol, GPT-judge baseline on identical data:

System	Accuracy	AUROC	Cost / latency
WITNESS + STAVE (default)	85.8%	0.844	$0, ~3 ms/decision
gpt-4o-mini (grounded judge)	86.3%	—	LLM call
gpt-3.5-turbo (HaluEval paper)	62.6%	—	LLM call
$0, zero-network verifier that statistically ties a strong LLM judge. Reproduce: python benchmarks/halueval_qa_faithful.py. Proof JSON.

Works with your stack
entroly wrap <agent> picks the best integration for each tool — proxy env-wrap for CLIs, auto-merged mcp.json for MCP-aware IDEs, or a copy-paste endpoint hint.

Wrap in one command: claude · cursor · codex · aider · gemini · windsurf · vscode · zed · cline · continue and 28 more.

Full agent list (38 targets)
As a library (LangChain, LlamaIndex, your own code):

from entroly import compress, compress_messages, optimize

compressed = compress(api_response, budget=2000)          # query-agnostic
messages   = compress_messages(messages, budget=30000)    # whole conversation
context    = optimize(fragments, budget=8000, query="fix the login bug")  # task-conditioned
In CI — fail the build if a prompt blows the token budget:

- run: pip install entroly && entroly batch --budget 8000 --fail-over-budget
When to use it · when to skip
Great fit

Large repos where the agent only sees a few files at a time
Chatty, multi-turn agents (cache alignment compounds the savings)
Anywhere you want answers checked against evidence before you trust them
Teams trying to cut a real, growing AI bill
Skip it (it'll just pass through)

Tiny repos or short prompts that already fit the budget
Judgment-heavy tasks where you want the full flagship model every time
What's inside
Most people install Entroly for input-token compression. It actually ships 19 local cost-saving mechanisms across input, inference, output, verification, and learning — each one readable in the source with a committed benchmark where applicable.

The 19 levers (and the file that implements each)
Engine & install options
WITNESS — check answers before you trust them
entroly witness --context-file evidence.txt --output-file answer.txt --mode strict
entroly proxy --witness strict --witness-profile rag    # suppress unsupported claims inline
Profiles tune false-positive behavior per workload (rag, qa, code fail closed; chat, summary warn). Every non-streaming response gets a proof certificate; the dashboard shows flagged claims, evidence snippets, and suppression counts. Optional offline DeBERTa NLI (ENTROLY_LOCAL_NLI=1) raises accuracy further at $0.

Compared to
Entroly	Compression tools	Top-K / RAG	Raw truncation
Approach	Rank → select → compress	Compress whatever's given	Embedding retrieval	Cut off
Token savings	70–95% (large repos)	50–70%	30–50%	0%
Quality loss	None measured	2–5%	Variable	High
Needs embeddings API	No	Varies	Yes	No
Reversible	Yes	Varies	Yes	No
Learns over time	Yes (PRISM)	No	No	No
Verifies the answer	Yes (WITNESS)	No	No	No
Compressing a bad selection is still a bad selection. Entroly ranks first, then compresses — so the model gets structure, not just fewer tokens.

Docs & community
Command reference
Architecture & full spec — Rust modules, 3-resolution compression, provenance, RAG comparison, SDK, LangChain.
For teams — ROI, security, deployment one-pager.
Limitations — where Entroly helps, where it passes through, and what it does not guarantee.
Cookbook — copy-paste recipes for common workflows.
Discussions · Issues
Apache-2.0 · local-first · no outbound analytics by default

pip install entroly && entroly go

