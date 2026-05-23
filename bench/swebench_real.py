#!/usr/bin/env python3
"""SWE-bench Lite - REAL repo retrieval (no answer leakage).

Replaces the synthetic harness's circular setup (audit P0-2). For each
sampled task we download the repository's tree **at the task's
base_commit** (GitHub codeload tarball — real files, no git history,
no patch content), index the real files with the shipped
`entroly_core` engine, query with the issue's `problem_statement`
(exactly what an agent gets), and measure whether the gold-modified
file paths — the only thing taken from the patch, as labels — are
retrieved. There is no reconstruction of file content from the answer.

Optional `--judge` (uses OPENAI_API_KEY): give gpt-4o-mini ONLY the
issue + the engine's top-K selected file list/snippets and ask which
file to edit; exact-match to a gold path is a leakage-free downstream
utility signal (the model never sees the patch).

Honest scope: this measures *retrieval* (did the engine surface the
files that must change), not task resolution. Sampling is stratified
across all 12 repos with a fixed seed — representative, not
cherry-picked. Repos that exceed the size/time guard are skipped and
reported, never silently dropped.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Prefer locally-built entroly_core.pyd (has recall_bm25)
_LOCAL_PYD = REPO_ROOT / "entroly" / "entroly_core.pyd"
if _LOCAL_PYD.exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location("entroly_core", str(_LOCAL_PYD))
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["entroly_core"] = _mod
        _spec.loader.exec_module(_mod)

SRC_EXT = (".py", ".pyx", ".pyi", ".rst", ".md", ".txt", ".cfg", ".toml",
           ".ini", ".yaml", ".yml")
_TARBALL_CACHE: dict[str, dict[str, str]] = {}

# ── Embedding baseline (MEASUREMENT ONLY — never ships) ───────────────
# text-embedding-3-small dense rerank of BM25's top-K — the realistic,
# cheap-but-strong neural bar Tier-0 must beat (this is how dense
# retrieval is actually deployed: retrieve-then-rerank, not brute dense
# over thousands of files). MEASUREMENT competitor, NOT a product
# dependency — the shipped path stays zero-dep / zero-network. One
# ~1000-char vector per candidate file, disk-cached by content hash so
# re-runs cost $0. Token spend kept minimal by design.
_EMB_MODEL = "text-embedding-3-small"
_EMB_CACHE_PATH = REPO_ROOT / "bench" / "_emb_cache.json"
_EMB_POOL = 150            # rerank BM25's top-N (fair strong deployment)
_EMB_HEAD_CHARS = 1000     # representative slice per file (~250 tokens)
_emb_cache: dict[str, list[float]] | None = None


def _emb_cache_load() -> dict[str, list[float]]:
    global _emb_cache
    if _emb_cache is None:
        try:
            _emb_cache = json.loads(_EMB_CACHE_PATH.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            _emb_cache = {}
    return _emb_cache


def _emb_cache_save() -> None:
    if _emb_cache is not None:
        try:
            _EMB_CACHE_PATH.write_text(json.dumps(_emb_cache), "utf-8")
        except Exception:  # noqa: BLE001
            pass


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed with disk cache keyed by sha256(text)."""
    import hashlib

    from openai import OpenAI
    cache = _emb_cache_load()
    keys = [hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()
            for t in texts]
    need = [(i, t) for i, (t, k) in enumerate(zip(texts, keys))
            if k not in cache]
    if need:
        client = OpenAI()
        B = 256
        for s in range(0, len(need), B):
            batch = need[s:s + B]
            for attempt in range(4):
                try:
                    resp = client.embeddings.create(
                        model=_EMB_MODEL,
                        input=[t for _, t in batch])
                    for (idx, _t), d in zip(batch, resp.data):
                        cache[keys[idx]] = d.embedding
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        for idx, _t in batch:
                            cache[keys[idx]] = []
                    else:
                        time.sleep(2 ** attempt)
        _emb_cache_save()
    return [cache.get(k, []) for k in keys]


def _cos(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def _embed_rank(files: dict[str, str], query: str, k: int,
                bm25_ranked: list[str]) -> list[str]:
    """Cheap-but-strong dense baseline: embed-rerank BM25's top
    `_EMB_POOL` candidates (one ~1000-char vector/file), cosine to the
    issue embedding. This is the realistic strong deployment of dense
    retrieval and bounds token spend to ~pool×250 tokens/task."""
    pool = bm25_ranked[:_EMB_POOL]
    if not pool:
        return []
    qv = _embed_texts([query[:4000]])[0]
    texts = [p + "\n" + files.get(p, "")[:_EMB_HEAD_CHARS] for p in pool]
    vecs = _embed_texts(texts)
    scored = sorted(
        zip(pool, vecs),
        key=lambda pv: (-_cos(qv, pv[1]), pv[0]),
    )
    out = [p for p, _ in scored]
    seen = set(out)
    for p in bm25_ranked:                # totality: BM25 order for the tail
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:k]


def _gold_files(patch: str) -> list[str]:
    """File paths the gold patch modifies - the labels (NOT content)."""
    out = []
    for m in re.finditer(r"^\+\+\+ b/(.+?)\s*$", patch, re.M):
        p = m.group(1).strip()
        if p and p != "dev/null":
            out.append(p)
    return sorted(set(out))


def _fetch_repo(repo: str, sha: str, max_mb: float,
                timeout: float) -> dict[str, str] | None:
    """Download repo tree at `sha`. Returns {relpath: content} or None
    if it exceeds the guard (honestly skipped)."""
    key = f"{repo}@{sha}"
    if key in _TARBALL_CACHE:
        return _TARBALL_CACHE[key]
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    try:
        t0 = time.time()
        raw = urllib.request.urlopen(url, timeout=timeout).read()
        if len(raw) > max_mb * 1e6:
            print(f"    [skip] {key}: {len(raw)/1e6:.0f}MB > {max_mb}MB guard")
            return None
        files: dict[str, str] = {}
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for m in tf.getmembers():
                if not m.isfile() or not m.name.endswith(SRC_EXT):
                    continue
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                try:
                    data = tf.extractfile(m)
                    if data is None:
                        continue
                    txt = data.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    continue
                if len(txt) > 20000:        # cap giant/generated files
                    txt = txt[:20000]
                files[rel] = txt
        print(f"    [ok]   {key}: {len(raw)/1e6:.1f}MB, "
              f"{len(files)} src files, {time.time()-t0:.1f}s")
        _TARBALL_CACHE[key] = files
        return files
    except Exception as e:  # noqa: BLE001
        print(f"    [err]  {key}: {type(e).__name__}: {e}")
        return None


def _stratified(ds, n: int, seed: int):
    import random
    rng = random.Random(seed)
    by_repo: dict[str, list] = defaultdict(list)
    for row in ds:
        by_repo[row["repo"]].append(row)
    repos = sorted(by_repo)
    picked = []
    per = max(1, n // len(repos))
    for r in repos:
        rows = by_repo[r]
        rng.shuffle(rows)
        picked.extend(rows[:per])
    rng.shuffle(picked)
    return picked[:n]


def _rank(engine_files: dict[str, str], query: str, k: int) -> list[str]:
    """Rank files by relevance using the engine's BM25+TPKS production path."""
    import entroly_core as ec
    eng = ec.EntrolyEngine()
    total_tokens = 0
    for path, content in engine_files.items():
        tc = max(1, len(content) // 4)
        eng.ingest(content, path, tc, False)
        total_tokens += tc

    ranked: list[str] = []
    seen: set[str] = set()
    try:
        if hasattr(eng, "recall_bm25"):
            # Fast BM25-only path (no knapsack overhead)
            for item in eng.recall_bm25(query, k):
                s = item.get("source", "")
                if s and s not in seen:
                    ranked.append(s)
                    seen.add(s)
        else:
            # Full optimize pipeline (same BM25 quality, more overhead)
            budget = total_tokens + 10000
            eng.advance_turn()
            result = eng.optimize(budget, query)
            selected = result.get("selected_fragments", []) or result.get("selected", [])
            for item in selected:
                s = item.get("source", "")
                if s and s not in seen:
                    ranked.append(s)
                    seen.add(s)
    except Exception:  # noqa: BLE001
        # Last resort: use recall (SimHash — poor quality but won't crash)
        for item in eng.recall(query, k):
            s = item.get("source", "")
            if s and s not in seen:
                ranked.append(s)
                seen.add(s)
    return ranked[:k] if len(ranked) > k else ranked


def _judge(model: str, issue: str, ranked: list[str],
           files: dict[str, str]) -> str | None:
    """Leakage-free downstream signal: model picks the file to edit from
    the engine's selection + issue. Never sees the patch."""
    from openai import OpenAI
    client = OpenAI()
    listing = "\n".join(
        f"- {p}\n    {files.get(p,'')[:200].strip()[:200]}"
        for p in ranked[:10]
    )
    msg = [
        {"role": "system", "content":
         "You are a senior engineer triaging a bug. Given an issue and a "
         "shortlist of candidate files (with the first lines of each), "
         "output ONLY the single file path most likely to need editing. "
         "Output the path verbatim, nothing else."},
        {"role": "user", "content":
         f"#Issue#\n{issue[:4000]}\n\n#Candidate files#\n{listing}\n\n"
         f"#File to edit#:"},
    ]
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, messages=msg, temperature=0.0, max_tokens=60)
            return (r.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            if attempt == 3:
                return None
            time.sleep(2 ** attempt)
    return None


def _load_env() -> None:
    f = REPO_ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"(?:export\s+)?OPENAI_API_KEY\s*=\s*"
                         r"[\"']?([^\"'\s]+)", line.strip())
            if m:
                os.environ["OPENAI_API_KEY"] = m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--max-mb", type=float, default=80.0)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--embed", action="store_true",
                    help="add text-embedding-3-large dense baseline "
                         "(MEASUREMENT ONLY; never ships; needs key)")
    ap.add_argument("--judge", action="store_true",
                    help="also run gpt-4o-mini leakage-free file-pick")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    kmax = max(ks)

    # Force offline — dataset is cached; prevents network-hang on HF hub
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    from entroly.localization import Tier0Localizer, _tok
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    tasks = _stratified(ds, args.samples, args.seed)

    methods = ["engine", "bm25", "tier0", "engine_s5", "tier0_ppr",
               "engine_s6"]
    if args.embed:
        _load_env()
        if os.environ.get("OPENAI_API_KEY"):
            methods.append("embed")
        else:
            print("  [warn] --embed set but no OPENAI_API_KEY; "
                  "skipping the embedding baseline")
            args.embed = False
    if args.judge:
        _load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            args.judge = False

    print("=" * 72)
    print(f"  SWE-bench Lite - REAL repo retrieval (n={len(tasks)}, "
          f"seed={args.seed}, stratified across all repos)")
    print("  Same-task head-to-head: engine(recall_bm25) | plain-BM25 | "
          "Tier0+RM3(zero-dep,SHIPS) | " + ("embed=" + _EMB_MODEL
          + "(MEASUREMENT ONLY, never ships)" if args.embed else
          "embed=OFF"))
    print("  PRE-REGISTERED PREDICTION: Tier0+RM3 hit@10 >= embedding "
          "hit@10 (a zero-dep method beats neural retrieval on code")
    print("  localization). If Tier0 < embed, the thesis is FALSIFIED - "
          "we report the gap and that justifies the air-gapped tier.")
    print("=" * 72)

    n_eval = skipped = 0
    acc = {m: {"hit": {k: 0 for k in ks},
               "frec": {k: 0.0 for k in ks}, "mrr": 0.0} for m in methods}

    for i, t in enumerate(tasks):
        gold = _gold_files(t["patch"])
        if not gold:
            continue
        files = _fetch_repo(t["repo"], t["base_commit"],
                            args.max_mb, args.timeout)
        if not files:
            skipped += 1
            continue
        gold = [g for g in gold if g in files]
        if not gold:
            skipped += 1
            continue
        query = t["problem_statement"] or t["instance_id"]
        gold_set = set(gold)
        loc = Tier0Localizer(files)
        bm25_full = loc._bm25_content.ranking(_tok(query))
        engine_full = _rank(files, query, max(kmax, 50))   # base for rerank
        engine_s5_full = loc.rerank(engine_full, query, kmax)
        ranked_by = {
            "engine": engine_full[:kmax],
            "bm25": bm25_full[:kmax],
            "tier0": loc.rank(query, kmax),                 # v4 pure fusion
            "engine_s5": engine_s5_full,                    # v4 rerank
            "tier0_ppr": loc.rerank_ppr(query, kmax),       # BM25 prior + PPR
            # engine_s6 = engine_s5 + deterministic edit-target prior
            # (window=20, frozen explicit cues, src>test>non-src class
            # re-prio with doc/test intent guards, test→source mirror).
            "engine_s6": loc.rerank_edit_target(
                engine_s5_full, query, kmax),
        }
        if "embed" in methods:
            ranked_by["embed"] = _embed_rank(files, query, kmax, bm25_full)
        n_eval += 1
        for m in methods:
            r = ranked_by[m]
            a = acc[m]
            for k in ks:
                topk = set(r[:k])
                if all(g in topk for g in gold):
                    a["hit"][k] += 1
                a["frec"][k] += sum(g in topk for g in gold) / len(gold)
            first = next((j for j, s in enumerate(r) if s in gold_set), None)
            a["mrr"] += 1.0 / (first + 1) if first is not None else 0.0
        if (i + 1) % 3 == 0:
            msg = "  ".join(
                f"{m}={acc[m]['hit'][10]/max(1,n_eval):.2f}" for m in methods)
            print(f"  [{i+1}/{len(tasks)}] eval={n_eval} hit@10  {msg}",
                  flush=True)

    print("\n" + "=" * 72)
    print(f"  REAL-REPO retrieval - evaluated {n_eval} tasks "
          f"({skipped} skipped: oversize/unavailable)")
    print("=" * 72)
    nz = max(1, n_eval)
    res = {"protocol": "SWE-bench Lite, real repo @ base_commit, "
                       "gold=patch file paths (labels only, no content)",
           "n_eval": n_eval, "n_skipped": skipped, "seed": args.seed,
           "embedding_model": _EMB_MODEL if args.embed else None,
           "methods": {}}
    for m in methods:
        a = acc[m]
        md: dict = {}
        line = [f"  {m:<7}"]
        for k in ks:
            h = a["hit"][k] / nz
            fr = a["frec"][k] / nz
            ci = 1.96 * (h * (1 - h) / nz) ** 0.5
            md[f"hit_rate@{k}"] = round(h, 4)
            md[f"hit_rate@{k}_ci95"] = round(ci, 4)
            md[f"file_recall@{k}"] = round(fr, 4)
            line.append(f"hit@{k}={h:.3f}±{ci:.3f} frec@{k}={fr:.3f}")
        md["mrr"] = round(a["mrr"] / nz, 4)
        line.append(f"mrr={md['mrr']:.3f}")
        res["methods"][m] = md
        print("  ".join(line))

    t10 = res["methods"]["tier0"]["hit_rate@10"]
    b10 = res["methods"]["bm25"]["hit_rate@10"]
    if "embed" in methods:
        e10 = res["methods"]["embed"]["hit_rate@10"]
        beat = t10 >= e10 - 1e-9
        verdict = (
            f"Tier0+RM3 hit@10={t10:.3f} vs embed({_EMB_MODEL})={e10:.3f}, "
            f"BM25={b10:.3f} -> "
            + ("THESIS HOLDS: a ZERO-DEP/zero-network method >= neural "
               "embeddings on real code localization. Bundled-ONNX tier "
               "is UNNECESSARY - ship Tier0." if beat else
               "THESIS FALSIFIED: embeddings beat Tier0 here. Report the "
               "gap honestly; THIS is the evidence that justifies the "
               "air-gapped include_bytes!+ort tier."))
    else:
        verdict = (f"Tier0+RM3 hit@10={t10:.3f} vs BM25={b10:.3f} "
                   "(embedding baseline not run)")
    res["verdict"] = verdict
    print("\n  " + verdict)
    out = REPO_ROOT / "bench" / "swebench_real_result.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out}")
    if args.json:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
