"""Falsification of the Fusion-4 "0.90 AUROC breakthrough" on HaluEval-QA.

Pre-registered hypothesis
-------------------------
The dominant signal G (entity-coverage-gap, frozen weight 0.80) does not
measure faithfulness — it exploits how HaluEval-QA *synthesizes* its two
answers: the correct answer is derived from the knowledge (entities
in-context) while the hallucinated answer is fabricated by entity
substitution (entities out-of-context). A cal/test split inside HaluEval
cannot detect this because both splits share the construction artifact.

Design (frozen weights W=.05 E=.05 G=.80 S=.10 from
results/fusion4_optimized.json — NOT re-fit here)
------------------------------------------------------------------------
For a fixed labelled sample, use gpt-4o-mini to break the artifact two
ways, then re-measure the SAME frozen fusion:

  C1  original            r            vs  h            (must ≈ 0.90)
  C2  entity-controlled   r            vs  h_ctrl        (h rewritten:
        still wrong/unsupported, but reuses ONLY entities present in the
        knowledge — removes the artifact from the positive class)
  C3  paraphrase-stress   r_para       vs  h            (r paraphrased:
        still correct & supported, but synonyms/abbreviations instead of
        verbatim — stresses the negative class against G's copy bias)
  C4  realistic           r_para       vs  h_ctrl        (both removed)

Pre-registered verdict rule (decide BEFORE seeing numbers)
----------------------------------------------------------
* ARTIFACT CONFIRMED if C2 AUROC drops >= 0.07 vs C1, OR mean G(h_ctrl)
  collapses to within 0.05 of mean G(r), OR C3 mean G(r_para) rises to
  within 0.05 of mean G(h) (faithful answers now look hallucinated).
* BREAKTHROUGH SURVIVES only if fusion AUROC stays >= 0.87 in ALL of
  C2/C3/C4. Anything between is "partially real, not the headline 0.90".

Outputs the per-condition fusion AUROC, the G-only and WITNESS-only
AUROCs, and the mean-signal table that mechanistically shows whether G
tracks construction or faithfulness.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 42
CAL = 2000  # match the optimizer's split; sample from TEST region
N_ITEMS = 400  # 400 items -> up to 4 conditions; gpt calls = 2*N
GPT_MODEL = "gpt-4o-mini"
WORKERS = 8
WEIGHTS = (0.05, 0.05, 0.80, 0.10)  # frozen: W, E, G, S
CACHE = Path(__file__).parent / "results" / "fusion4_falsification_cache.json"

# EXACT entity patterns from fusion4_weight_optimizer.py:32 — apples-to-apples.
EP = [re.compile(r"\b\d+\.?\d*\b"), re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")]


def _entity_gap(ctx: str, ans: str) -> float:
    """Byte-identical to optimizer lines 40-43."""
    ae: set[str] = set()
    cl = ctx.lower()
    for p in EP:
        for m in p.finditer(ans):
            ae.add(m.group().lower())
    return (sum(1 for e in ae if e not in cl) / max(len(ae), 1)) if ae else 0.0


def auroc(scores: list[float], labels: list[int]) -> float:
    p = sorted(zip(scores, labels))
    r = [0.0] * len(p)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[j + 1][0] == p[i][0]:
            j += 1
        a = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[k] = a
        i = j + 1
    n1 = sum(y for _, y in p)
    n0 = len(p) - n1
    if n0 == 0 or n1 == 0:
        return 0.5
    return (sum(rr for rr, (_, y) in zip(r, p) if y == 1) - n1 * (n1 + 1) / 2) / (
        n0 * n1
    )


def _load_env() -> None:
    f = Path(__file__).resolve().parent.parent / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(
                r"(?:export\s+)?OPENAI_API_KEY\s*=\s*"
                r"[\"']?([^\"'\s]+)",
                line.strip(),
            )
            if m:
                os.environ["OPENAI_API_KEY"] = m.group(1)


_H_CTRL_SYS = (
    "You rewrite an answer so it is STILL factually wrong or unsupported "
    "given the knowledge, but uses ONLY names, numbers, and proper nouns "
    "that appear verbatim in the provided knowledge. Introduce NO new "
    "named entity or number that is absent from the knowledge. Keep it a "
    "fluent, plausible-looking 1-2 sentence answer to the question. "
    "Output only the rewritten answer."
)
_R_PARA_SYS = (
    "You paraphrase a CORRECT answer so it stays factually correct and "
    "fully supported by the knowledge, but changes surface form: prefer "
    "synonyms, expand or contract abbreviations, write numbers as words "
    "(or vice versa), and restructure the sentence. Do NOT copy long noun "
    "phrases verbatim if a faithful paraphrase exists. Output only the "
    "paraphrased answer."
)


def _gpt(client, system: str, user: str) -> str:
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=120,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            if attempt == 3:
                return ""
            time.sleep(2**attempt)
    return ""


def build_dataset() -> list[dict]:
    if CACHE.exists():
        print(f"  Loading cached constructed answers: {CACHE}")
        return json.loads(CACHE.read_text(encoding="utf-8"))

    from datasets import load_dataset
    from openai import OpenAI

    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    items = [
        (
            str(r.get("knowledge", "")),
            str(r.get("question", "")),
            str(r.get("right_answer", "")),
            str(r.get("hallucinated_answer", "")),
        )
        for r in ds
        if all(
            r.get(f)
            for f in ("knowledge", "question", "right_answer", "hallucinated_answer")
        )
    ]
    random.Random(SEED).shuffle(items)
    sample = items[CAL * 2 : CAL * 2 + N_ITEMS]  # TEST region only

    client = OpenAI()
    print(
        f"  Constructing artifact-broken answers with {GPT_MODEL} "
        f"({2 * len(sample)} calls)...",
        flush=True,
    )

    def work(it):
        k, q, ra, ha = it
        kq = f"Knowledge: {k}\nQuestion: {q}\n"
        h_ctrl = _gpt(client, _H_CTRL_SYS, kq + f"Wrong answer to rewrite: {ha}")
        r_para = _gpt(client, _R_PARA_SYS, kq + f"Correct answer to paraphrase: {ra}")
        return {
            "k": k,
            "q": q,
            "ra": ra,
            "ha": ha,
            "h_ctrl": h_ctrl or ha,
            "r_para": r_para or ra,
        }

    out: list[dict] = [None] * len(sample)  # type: ignore
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(work, it): i for i, it in enumerate(sample)}
        done = 0
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 100 == 0:
                print(
                    f"    {done}/{len(sample)} ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Cached: {CACHE}")
    return out


def main() -> None:
    print("=" * 74)
    print("  Fusion-4 FALSIFICATION — is 0.90 a HaluEval entity-swap artifact?")
    print("=" * 74)
    _load_env()
    rows = build_dataset()

    from entroly.ravs.ece import compute_fisher_curvature
    from entroly.ravs.spectral import compute_spectral_consistency
    from entroly.witness import WitnessAnalyzer

    az = WitnessAnalyzer(use_nli=False, force_python=True, profile="benchmark_qa")

    def signals(ctx: str, ans: str):
        w = 1 - float(az.analyze(ctx, ans).summary_score)
        mk, _, _ = compute_fisher_curvature(ans)
        e = min(1, mk * 2.5)
        g = _entity_gap(ctx, ans)
        s = 1 - compute_spectral_consistency(ctx, ans).score
        return w, e, g, s

    ww, we, wg, ws = WEIGHTS

    def fuse(sig):
        w, e, g, s = sig
        return min(1, max(0, ww * w + we * e + wg * g + ws * s))

    # Compute signals once per answer variant.
    R, H, HC, RP = [], [], [], []
    gR = gH = gHC = gRP = 0.0
    t0 = time.perf_counter()
    for idx, r in enumerate(rows):
        ctx = f"{r['k']}\n\nQuestion: {r['q']}"
        sr = signals(ctx, r["ra"])
        sh = signals(ctx, r["ha"])
        shc = signals(ctx, r["h_ctrl"])
        srp = signals(ctx, r["r_para"])
        R.append(sr)
        H.append(sh)
        HC.append(shc)
        RP.append(srp)
        gR += sr[2]
        gH += sh[2]
        gHC += shc[2]
        gRP += srp[2]
        if (idx + 1) % 100 == 0:
            print(
                f"    signals {idx + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    n = len(rows)

    def cond(neg, pos, name):
        sc = [fuse(x) for x in neg] + [fuse(x) for x in pos]
        lb = [0] * len(neg) + [1] * len(pos)
        gg = [x[2] for x in neg] + [x[2] for x in pos]
        wwl = [x[0] for x in neg] + [x[0] for x in pos]
        return {
            "name": name,
            "fusion": auroc(sc, lb),
            "g_only": auroc(gg, lb),
            "witness_only": auroc(wwl, lb),
        }

    C1 = cond(R, H, "C1 original (r vs h)")
    C2 = cond(R, HC, "C2 entity-controlled (r vs h_ctrl)")
    C3 = cond(RP, H, "C3 paraphrase-stress (r_para vs h)")
    C4 = cond(RP, HC, "C4 realistic (r_para vs h_ctrl)")

    mgR, mgH, mgHC, mgRP = gR / n, gH / n, gHC / n, gRP / n

    print(f"\n  n={n} items per class | frozen weights W={ww} E={we} G={wg} S={ws}\n")
    print(f"  {'condition':<34}{'fusion':>8}{'G-only':>8}{'WIT':>7}")
    print("  " + "-" * 56)
    for c in (C1, C2, C3, C4):
        print(
            f"  {c['name']:<34}{c['fusion']:>8.4f}"
            f"{c['g_only']:>8.4f}{c['witness_only']:>7.4f}"
        )
    print("\n  Mean entity-gap G by answer type (the mechanism):")
    print(f"    faithful r        = {mgR:.4f}")
    print(f"    hallucinated h    = {mgH:.4f}   (orig positive)")
    print(f"    entity-ctrl h_ctrl= {mgHC:.4f}   (artifact removed)")
    print(f"    paraphrased r_para= {mgRP:.4f}   (faithful, non-verbatim)")

    drop_c2 = C1["fusion"] - C2["fusion"]
    hc_collapse = abs(mgHC - mgR) <= 0.05
    rp_inflate = abs(mgRP - mgH) <= 0.05 or mgRP >= 0.5 * mgH + 0.5 * mgR
    artifact = (drop_c2 >= 0.07) or hc_collapse or rp_inflate
    survives = all(c["fusion"] >= 0.87 for c in (C2, C3, C4))

    print("\n" + "=" * 74)
    print("  PRE-REGISTERED VERDICT")
    print("=" * 74)
    print(f"  C1->C2 fusion drop      = {drop_c2:+.4f}  (artifact if >= +0.07)")
    print(f"  G(h_ctrl)~G(r) collapse = {hc_collapse}  (|{mgHC:.3f}-{mgR:.3f}|<=0.05)")
    print(f"  G(r_para) inflated      = {rp_inflate}")
    print(f"  Fusion>=0.87 in C2,C3,C4= {survives}")
    if artifact and not survives:
        verdict = (
            "ARTIFACT CONFIRMED — the 0.90 is largely HaluEval's "
            "entity-swap synthesis. Does NOT generalize. Keep "
            "README at WITNESS 0.798."
        )
    elif survives and not artifact:
        verdict = (
            "BREAKTHROUGH SURVIVES — fusion holds with the "
            "artifact removed. Defensible to report."
        )
    else:
        verdict = (
            "PARTIAL — real signal but inflated; the headline "
            "0.90 is not defensible. Report the artifact-broken "
            "number, not 0.90."
        )
    print(f"\n  >>> {verdict}\n")

    res = {
        "weights": {"W": ww, "E": we, "G": wg, "S": ws},
        "n": n,
        "conditions": [C1, C2, C3, C4],
        "mean_G": {"r": mgR, "h": mgH, "h_ctrl": mgHC, "r_para": mgRP},
        "drop_c1_c2": drop_c2,
        "artifact": artifact,
        "survives": survives,
        "verdict": verdict,
    }
    op = Path(__file__).parent / "results" / "fusion4_falsification.json"
    op.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  Saved: {op}")


if __name__ == "__main__":
    main()
