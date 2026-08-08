#!/usr/bin/env python3
"""Does evidence delivery predict answer correctness? (Q-A, first slice)

Every result in the graph-lane work is a *proxy*: it measures whether a file was
delivered, never whether a model answered correctly. If the proxy does not
predict correctness, then those numbers -- including the D-REJECTED verdict on
graph-aware selection -- are not safe to act on. This experiment tests the
proxy against a real model before any more weight is put on it.

PREREGISTERED, written before the first model call:

  H1  Indirect recall predicts answer correctness. Arms delivering the callee
      file more often answer correctly more often, and the ordering of arms by
      correctness matches their ordering by indirect recall
      (qccr > bm25 > hcc > null).

  Verdict rules, fixed here:
    * VOID if the null arm scores > 0.10. The question would be answerable
      from the prompt alone and nothing else can be concluded.
    * SUPPORTED if qccr's correctness exceeds hcc's by >= 0.20 -- the proxy
      said 76.7% vs 3.3% delivery, so a real gap must appear.
    * REFUTED if qccr does not beat hcc. The proxy would then be measuring
      something that does not reach the model, and the Q-B verdict must be
      reopened rather than defended.

Task and gold answer
--------------------
Reuses the import-resolved tasks from `graph_lane_quality.py`: caller `S` in
file `A` imports and calls `T`, defined in file `B`. The question asks for
`T`'s *parameter names* -- a fact recorded only in `B`. `A` contains the call
site, not the signature.

Two leakage filters, applied before any model call:
  * tasks whose call site in `A` uses keyword arguments are dropped, because a
    keyword reveals a parameter name without `B`;
  * `T` must have >= 2 parameters after dropping `self`/`cls`, so the answer is
    not guessable.

Scoring is exact set equality on parameter names. No partial credit, no judge
model -- a fuzzy scorer here would be the benchmark theater this programme
exists to avoid.

FIRST RUN: VOID -- the model, not the context, was the bottleneck
----------------------------------------------------------------
Executed against a local `qwen2.5-coder:1.5b` via Ollama (12 probes, budget
6000, 0 errors):

    null 0/12 | oracle 4/12 | bm25 1/12 | qccr 0/12 | hcc 0/12

**The oracle arm scored 4/12.** Oracle hands the model the single file that
defines the callee, so it is the capability control: if the model cannot answer
with the evidence directly in front of it, no arm below it is interpretable.
At 33% the 1.5B model is the limiting factor, and reading `qccr 0/12` as a
statement about selection quality would be measuring the model and blaming the
compressor -- the exact error this file exists to prevent.

The run is therefore recorded as VOID rather than as a result. The same model
answered a 50-token version of the question correctly, so the failure is
context-length capability, not prompt format.

SECOND RUN: also VOID -- compute, and biased in a way that matters
------------------------------------------------------------------
The 7B re-run (`entroly-qwen2.5-7b-32k`, digest 6561cc69dc5a) failed 49 of 60
calls with `TimeoutError` at a 300 s per-call limit. The failures are not
random:

    arm      completed  timed out
    null             7          5     (0 context tokens)
    oracle           3          9     (5,425)
    bm25             1         11     (5,935)
    qccr             0         12     (5,363)
    hcc              0         12     (5,500)

**Timeout rate tracks context size, so every large-context arm completed zero
calls.** A run with that structure does not merely have few samples; it is
biased toward whichever arm sends least, which is the opposite of what this
experiment is for. Reporting `qccr 0/12` from it would be reporting a
stopwatch, not a selector.

Both attempts are therefore void, for different reasons: the 1.5B model was
capability-limited (oracle 4/12), the 7B was compute-limited (49/60 timeouts).
Q-A remains unanswered.

What would actually settle it, in order of preference:
  * a GPU -- Ollama reports `size_vram 0.0` here, so all inference was CPU;
  * a working hosted API key (`--backend openai`), the path this harness was
    written for;
  * a smaller context budget, which changes the experiment rather than running
    it, and should be a last resort.

Raising the timeout alone is not a fix: 60 calls already exceeding 300 s each
is hours of wall clock for one 12-probe matrix.

THIRD RUN: VALID -- and H1 is REFUTED -- **OBSERVED**
-----------------------------------------------------
Reproduce:

    python benchmarks/answer_correctness_bridge.py --limit 5 --budget 2000 \
        --backend ollama --model entroly-qwen2.5-7b-32k:latest \
        --max-oracle-tokens 2700 --timeout 290

`entroly-qwen2.5-7b-32k` (digest 6561cc69dc5a), 5 probes, budget 2000, **0
errors**:

    arm      correct   ctx tokens
    null       0/5              0
    oracle     5/5          1,678
    bm25       0/5          1,941
    qccr       1/5          2,063
    hcc        1/5          1,730

Both gates pass. The null arm scores 0.0, below the 0.10 void threshold, so the
question is not answerable from the prompt. The oracle arm scores 5/5, so the
model can perform the task exactly when the defining file is present -- every
selection failure below is a selection failure, not a model failure.

**Verdict: H1 REFUTED.** The preregistered rule required qccr to exceed hcc by
>= 0.20; both scored 0.20, so the rule's REFUTED branch applies, which states:
*the proxy would then be measuring something that does not reach the model, and
the Q-B verdict must be reopened rather than defended.*

That reopening is now owed. §Q-B rejected graph-aware selection on **delivery**
-- qccr 76.7% indirect recall against hcc 3.3%. This run shows that delivery
advantage producing **no** correctness advantage at a 2,000-token budget. The
D-REJECTED verdict rested on a proxy that this measurement does not support.

The sharper observation, and the one that survives the small sample: **oracle
5/5 against every selection arm at 1/5 or 0/5.** The model answers perfectly
with the defining file and almost never with any arm's rendering of it at this
budget. Selection delivered usable evidence in 2 of 20 arm-probes.

Every miss is `UNKNOWN`, never a wrong signature. The model reported absent
evidence honestly rather than fabricating, so the failure is delivery, not
hallucination. That also means the arms are being scored on whether the
signature *survived* their rendering -- qccr delivers the file as extracted
sentences, and the signature is frequently not among them. This is the
"strong locator, weak reader" pattern seen in dogfooding, now confirmed with a
model in the loop.

**Power is the honest limit.** n=5 cannot separate qccr from hcc; 1/5 against
1/5 is one probe. What n=5 *does* support is the 5/5-versus-≤1/5 contrast,
which is not marginal. Treat the REFUTED verdict as "the delivery proxy is not
validated" rather than "the arms are proven equal".
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import urllib.request
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.graph_lane_quality import (
    REPO,
    Task,
    _bm25_rank,
    _fill,
    _pool_for,
    _read,
    _tokens,
    _tracked_python_files,
    _arm_hcc,
    _arm_qccr,
)

# `raw_full` is deliberately absent. Sending all 48 pool files exceeds the
# model's context window and costs real money to learn nothing: the delivery
# benchmark already established that the answer is present in the pool.
# `oracle` -- the callee file alone -- is the better ceiling, because it tests
# the question's answerability rather than the pool's completeness.
ARMS = ("null", "oracle", "bm25", "qccr", "hcc")


@dataclass(frozen=True)
class Probe:
    task: Task
    params: tuple[str, ...]


def _signature_params(args: ast.arguments) -> tuple[str, ...]:
    """Every named parameter, in signature order.

    Reading only `args.args` silently dropped positional-only and keyword-only
    parameters, which made the gold answer *incomplete* rather than merely
    strict. Measured case: `slot_substitution_score(claim_text, evidence_text,
    *, window, align_threshold)` produced a gold of two names, so a model that
    correctly answered all four was scored wrong. A benchmark that penalises
    correct answers measures its own defect.

    `*args`/`**kwargs` are excluded deliberately: they are not names a caller
    can be expected to recite, and including them would make the task about
    Python syntax rather than about whether the evidence reached the model.
    """
    named = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    return tuple(a.arg for a in named if a.arg not in {"self", "cls"})


def _callee_params(callee: str, callee_file: str) -> tuple[str, ...] | None:
    """Parameter names of `callee` as defined in `callee_file`.

    Classes are deliberately not resolved. Asking for "the parameter names of
    `CacheAligner`" is ambiguous: a model that answers with the parameters of
    its `align()` method rather than its `__init__` has not made an error, and
    that ambiguity was observed scoring a defensible answer as a miss.
    """
    try:
        tree = ast.parse((REPO / callee_file).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == callee:
            return _signature_params(node.args)
    return None


def _call_uses_keywords(caller: str, caller_file: str, callee: str) -> bool:
    """True when the call site names a parameter, leaking the answer."""
    try:
        tree = ast.parse((REPO / caller_file).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return True  # unparseable: exclude rather than risk leakage
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != caller:
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == callee
                and any(kw.arg for kw in sub.keywords)
            ):
                return True
    return False


def build_probes(
    tasks: list[Task], limit: int, max_oracle_tokens: int = 0
) -> list[Probe]:
    """Probes whose gold answer is unambiguous and whose oracle arm is runnable.

    `max_oracle_tokens` bounds the callee file, which is what the oracle arm
    sends whole. It exists because the capability control has to actually
    complete: on CPU inference the 7B model answered 5/5 on callee files under
    2,700 tokens, and timed out on larger ones. Bounding the file keeps the
    oracle meaningful (still the entire defining file) while keeping the run
    inside its timeout. 0 disables the bound.
    """
    probes: list[Probe] = []
    for task in tasks:
        params = _callee_params(task.callee, task.callee_file)
        if not params or len(params) < 2:
            continue
        if _call_uses_keywords(task.caller, task.caller_file, task.callee):
            continue
        if max_oracle_tokens:
            try:
                size = len((REPO / task.callee_file).read_text(
                    encoding="utf-8", errors="replace")) // 4
            except OSError:
                continue
            if size > max_oracle_tokens:
                continue
        probes.append(Probe(task=task, params=params))
        if len(probes) >= limit:
            break
    return probes


def _context_for(arm: str, probe: Probe, pool: list[str],
                 texts: dict[str, str], budget: int) -> str:
    """The text an arm actually emits -- not the full files it touched.

    An earlier version rebuilt context by joining the whole source of every
    file an arm selected. That discarded the compression: qccr emitted 91,250
    tokens against a 2,000-token budget, because it delivers extracted spans
    while this reconstructed the originals. The arms were not budget-matched,
    which would have made any correctness comparison meaningless.
    """
    if arm == "null":
        return ""

    if arm == "oracle":
        rel = probe.task.callee_file
        return f"### {rel}\n{texts.get(rel, _read(rel))}"

    if arm == "bm25":
        # Selects whole files, so the file text IS what it emits.
        chosen = _fill(_bm25_rank(pool, texts, probe.task.query), texts, budget)
        return "\n\n".join(f"### {rel}\n{texts[rel]}" for rel in sorted(chosen))

    if arm == "qccr":
        from entroly.qccr import select as qccr_select

        frags = [
            {"id": f"file:{rel}", "source": f"file:{rel}", "content": texts[rel],
             "token_count": _tokens(texts[rel]), "relevance": 0.5}
            for rel in pool
        ]
        picked = qccr_select(frags, token_budget=budget, query=probe.task.query)
        return "\n\n".join(
            f"### {str(f.get('source', '')).removeprefix('file:')}\n{f.get('content', '')}"
            for f in picked
        )

    # hcc: level 2 is a skeleton and level 3 is full content; emit each as the
    # engine rendered it, at the resolution it charged itself for.
    from benchmarks.engine_isolation import assert_engine_isolated, isolated_engine_dir

    with isolated_engine_dir():
        assert_engine_isolated()
        from entroly_core import EntrolyEngine

        engine = EntrolyEngine()
        for rel in pool:
            engine.ingest(texts[rel], f"file:{rel}", _tokens(texts[rel]), False)
        result = engine.hierarchical_compress(budget, probe.task.query, None)
        if not isinstance(result, dict):
            return ""
        parts: list[str] = []
        # `level2_cluster` is the rendered skeleton text. The `level2_fragments`
        # entries carry each fragment's FULL content with its full token_count,
        # while HCC charges itself only the skeleton cost (380 vs 2,541 tokens
        # on one measured file) -- so joining their content would emit several
        # times the budget the engine believed it spent.
        cluster = str(result.get("level2_cluster") or "").strip()
        if cluster:
            parts.append(cluster)
        for frag in result.get("level3_fragments") or []:
            src = str(frag.get("source", "")).removeprefix("file:")
            parts.append(f"### {src}\n{frag.get('content', '')}")
        return "\n\n".join(parts)


# Deliberately small. The arms differ only in context, so the question must be
# answerable by a modest local model whenever the evidence is present -- a task
# that is hard for reasons other than retrieval measures the model, not the
# context. Two earlier faults are fixed here: it asked for parameters "in order"
# while the scorer compares sets, and it made the model first reason from caller
# to callee before answering. The model only has to find `def <callee>(...)`.
_PROMPT = """{context_block}
What are the parameter names of the function `{callee}`?
Answer with only the names separated by commas. If it is not shown above, answer: UNKNOWN"""


def _ollama_post(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ollama_identity(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    """Pin which weights answered, so the result is reproducible later."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout) as response:
        tags = json.loads(response.read().decode("utf-8"))
    matches = [v for v in tags.get("models", []) if v.get("name") == model]
    if not matches:
        raise RuntimeError(f"Ollama model {model!r} is not installed")
    return {
        "name": model,
        "digest": matches[0].get("digest"),
        "details": matches[0].get("details", {}),
    }


def _ask_ollama(base_url: str, model: str, prompt: str, timeout: float) -> str:
    """Greedy, seeded decoding -- the arms must differ only in context."""
    response = _ollama_post(
        base_url,
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_predict": 60},
        },
        timeout,
    )
    return str(response.get("response", "")).strip()


def _ask(client: Any, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _score(reply: str, gold: tuple[str, ...]) -> bool:
    if not reply or reply.strip().upper().startswith("UNKNOWN"):
        return False
    got = [p.strip().strip("`'\"") for p in reply.replace("\n", ",").split(",")]
    got = [p for p in got if p]
    return set(got) == set(gold)


def run(limit: int, budget: int, model: str, seed: int,
        backend: str = "ollama", base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0, max_oracle_tokens: int = 0) -> dict[str, Any]:
    client = None
    identity: dict[str, Any] = {}
    if backend == "ollama":
        identity = _ollama_identity(base_url, model, 30.0)
    else:
        from openai import OpenAI

        client = OpenAI()
    payload = json.loads(
        (REPO / "benchmarks" / "results" / "graph_lane_tasks.json").read_text(encoding="utf-8")
    )
    tasks = [Task(**t) for t in payload["tasks"]]
    probes = build_probes(tasks, limit, max_oracle_tokens)
    corpus = sorted(
        {str(p.relative_to(REPO)).replace("\\", "/") for p in _tracked_python_files()}
    )

    records: list[dict[str, Any]] = []
    for probe in probes:
        pool = _pool_for(probe.task, corpus, 48, seed)
        texts = {rel: _read(rel) for rel in pool}
        for arm in ARMS:
            context = _context_for(arm, probe, pool, texts, budget)
            block = f"Codebase context:\n{context}\n" if context else "No codebase context is available.\n"
            prompt = _PROMPT.format(
                context_block=block,
                caller=probe.task.caller,
                callee=probe.task.callee,
            )
            try:
                if backend == "ollama":
                    reply = _ask_ollama(base_url, model, prompt, timeout)
                else:
                    reply = _ask(client, model, prompt)
                error = ""
            except Exception as exc:  # noqa: BLE001
                reply, error = "", f"{type(exc).__name__}: {exc}"
            records.append({
                "caller": probe.task.caller,
                "callee": probe.task.callee,
                "callee_file": probe.task.callee_file,
                "arm": arm,
                "gold": list(probe.params),
                "reply": reply[:200],
                "correct": _score(reply, probe.params),
                "context_tokens": _tokens(context) if context else 0,
                "error": error,
            })
            time.sleep(0.05)

    return {
        "pinned_ref": payload["pinned_ref"],
        "backend": backend,
        "model": model,
        "model_identity": identity,
        "budget": budget,
        "probes": len(probes),
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--backend", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--model", default=os.environ.get("BRIDGE_MODEL", "entroly-qwen2.5-7b-32k:latest"))
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-oracle-tokens", type=int, default=0,
                    help="bound the callee file the oracle arm sends whole")
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "answer_correctness_bridge.json")
    args = ap.parse_args()

    payload = run(args.limit, args.budget, args.model, args.seed,
                  backend=args.backend, base_url=args.base_url, timeout=args.timeout,
                  max_oracle_tokens=args.max_oracle_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    errors = [r for r in payload["records"] if r["error"]]
    print(f"probes {payload['probes']}  model {payload['model']}  errors {len(errors)}")
    if errors:
        print("first error:", errors[0]["error"][:180])
    print(f"\n  {'arm':<10}{'correct':>10}{'ctx tokens':>12}")
    for arm in ARMS:
        rows = [r for r in payload["records"] if r["arm"] == arm]
        if not rows:
            continue
        hit = sum(1 for r in rows if r["correct"])
        avg = sum(r["context_tokens"] for r in rows) / len(rows)
        print(f"  {arm:<10}{hit:>4}/{len(rows):<5}{avg:>12.0f}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
