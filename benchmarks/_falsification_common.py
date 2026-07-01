"""
Generic C1-C4 Falsification Protocol
=====================================

Generalizes the Fusion-4 falsification methodology (benchmarks/fusion4_falsification.py)
to any (context, query, right_answer, hallucinated_answer) dataset. Used by the
per-benchmark adapters under Phase 0 step 1.1 of the EICV trust push.

Protocol (per EICV_PREREGISTRATION.md §4)
-----------------------------------------
  C1 — original           (r vs h)              headline AUROC
  C2 — entity-controlled  (r vs h_ctrl)         strips entity-coverage artifact
  C3 — paraphrase-stress  (r_para vs h)         strips verbatim-copy artifact
  C4 — realistic          (r_para vs h_ctrl)    both stripped

Backends
--------
  deterministic — rule-based entity-swap + number/word swap + simple paraphrase.
                  Free, reproducible, weaker than GPT (so failed-falsification
                  here is conservative; surviving here is strong evidence).
  gpt-4o-mini   — adversarial GPT-backed. Reuses fusion4_falsification.py prompts.
                  Caches to disk. Requires OPENAI_API_KEY.

Reporting rule (per EICV_PREREGISTRATION.md §4)
-----------------------------------------------
  trust_defensible iff backend produced clean C1..C4 AND min(C1..C4) >= target - 0.03
  artifact_drop = C1 - C4; report always
  backend is recorded in the output JSON

Fusion classifier
-----------------
Same frozen weights as Fusion-4: W=.05, E=.05, G=.80, S=.10
(WITNESS, ECE fisher curvature, entity gap, spectral consistency).
This keeps the per-benchmark probe apples-to-apples with the HaluEval-QA result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make the entroly package importable when these scripts are run directly.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("entroly.benchmarks.falsification")

# Frozen Fusion-4 weights: keep apples-to-apples with fusion4_falsification.py
WEIGHTS = (0.05, 0.05, 0.80, 0.10)  # W, E, G, S

# Entity patterns matching fusion4_weight_optimizer.py:32 byte-for-byte.
_ENTITY_PATTERNS = [
    re.compile(r"\b\d+\.?\d*\b"),
    re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"),
]


@dataclass(frozen=True)
class FalsItem:
    """One labelled item from a benchmark."""

    context: str  # the shared evidence (knowledge / doc / dialogue history)
    query: str  # the question (or empty string for tasks without one)
    right: str  # grounded answer
    halu: str  # hallucinated answer
    item_id: str = ""  # benchmark-specific stable identifier (optional)


@dataclass
class FalsRecord:
    """One processed item with all four answer variants."""

    item_id: str
    context: str
    query: str
    right: str
    halu: str
    h_ctrl: str
    r_para: str
    backend: str  # "deterministic" or "gpt-4o-mini"


# ── Scoring (mirrors fusion4_falsification.py:202-216) ──────────────────


def _entity_gap(ctx: str, ans: str) -> float:
    """G: fraction of answer-entities NOT in the context. Byte-identical to
    fusion4_weight_optimizer.py:40-43 so the falsification probe is apples-to-
    apples with the HaluEval-QA fusion4 result."""
    ae: set[str] = set()
    cl = ctx.lower()
    for p in _ENTITY_PATTERNS:
        for m in p.finditer(ans):
            ae.add(m.group().lower())
    return (sum(1 for e in ae if e not in cl) / max(len(ae), 1)) if ae else 0.0


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U AUROC with tie correction. Identical to
    fusion4_falsification.py:76-93."""
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


def _signals(analyzer: Any, ctx: str, ans: str) -> tuple[float, float, float, float]:
    """The four-signal vector (W, E, G, S) used by Fusion-4."""
    from entroly.ravs.ece import compute_fisher_curvature
    from entroly.ravs.spectral import compute_spectral_consistency

    w = 1.0 - float(analyzer.analyze(ctx, ans).summary_score)
    mk, _, _ = compute_fisher_curvature(ans)
    e = min(1.0, mk * 2.5)
    g = _entity_gap(ctx, ans)
    s = 1.0 - compute_spectral_consistency(ctx, ans).score
    return w, e, g, s


def _fuse(sig: tuple[float, float, float, float]) -> float:
    ww, we, wg, ws = WEIGHTS
    w, e, g, s = sig
    return min(1.0, max(0.0, ww * w + we * e + wg * g + ws * s))


# ── Deterministic backend ───────────────────────────────────────────────


_NUMBER_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}
_WORD_TO_NUMBER = {v: k for k, v in _NUMBER_TO_WORD.items()}

# Conservative synonym swaps — only safe drop-ins that don't change meaning.
# Used in deterministic r_para to break verbatim-copy artifacts without
# introducing semantic drift. Curated from common QA answer verbs.
_SYNONYM_SWAPS = [
    ("started", "began"),
    ("began", "started"),
    ("created", "founded"),
    ("founded", "established"),
    ("established", "founded"),
    ("located", "situated"),
    ("situated", "located"),
    ("known as", "called"),
    ("called", "named"),
    ("received", "won"),
    ("won", "received"),
    ("authored", "wrote"),
    ("wrote", "authored"),
    ("formed", "founded"),
    ("primarily", "mainly"),
    ("mainly", "primarily"),
    ("approximately", "around"),
    ("around", "approximately"),
    ("often", "frequently"),
    ("frequently", "often"),
    ("important", "significant"),
    ("significant", "important"),
    ("various", "several"),
    ("several", "various"),
    ("commonly", "typically"),
    ("typically", "commonly"),
]


def _extract_entities(text: str) -> list[tuple[int, int, str, str]]:
    """Return (start, end, token, type) for each entity in text. type is
    'number' or 'proper'."""
    spans: list[tuple[int, int, str, str]] = []
    for m in _ENTITY_PATTERNS[0].finditer(text):
        spans.append((m.start(), m.end(), m.group(), "number"))
    for m in _ENTITY_PATTERNS[1].finditer(text):
        spans.append((m.start(), m.end(), m.group(), "proper"))
    spans.sort()
    # Drop overlaps (longer wins)
    out: list[tuple[int, int, str, str]] = []
    for s in spans:
        if out and s[0] < out[-1][1]:
            if s[1] - s[0] > out[-1][1] - out[-1][0]:
                out[-1] = s
            continue
        out.append(s)
    return out


def deterministic_h_ctrl(
    knowledge: str, hallucinated: str, *, rng: random.Random
) -> str:
    """Entity-controlled hallucination rewrite. Replace each out-of-knowledge
    entity in `hallucinated` with a random in-knowledge entity of the same
    type. Keeps wrongness (because the structure is unchanged) but strips the
    "uses entity NOT in knowledge" artifact that the G signal exploits.

    Falls back to original hallucinated text if no in-knowledge entities of the
    needed type exist (small fraction of items)."""
    if not hallucinated.strip():
        return hallucinated
    k_lower = knowledge.lower()
    k_entities = _extract_entities(knowledge)
    k_by_type: dict[str, list[str]] = {"number": [], "proper": []}
    for _, _, tok, t in k_entities:
        if tok not in k_by_type[t]:
            k_by_type[t].append(tok)
    if not k_by_type["number"] and not k_by_type["proper"]:
        return hallucinated

    out = []
    last = 0
    h_entities = _extract_entities(hallucinated)
    for s, e, tok, t in h_entities:
        out.append(hallucinated[last:s])
        if tok.lower() in k_lower:
            out.append(tok)
        else:
            choices = k_by_type[t]
            if choices:
                # Deterministic-but-shuffled replacement
                replacement = rng.choice(choices)
                out.append(replacement)
            else:
                out.append(tok)
        last = e
    out.append(hallucinated[last:])
    return "".join(out)


def deterministic_r_para(knowledge: str, right: str, *, rng: random.Random) -> str:
    """Rule-based paraphrase of a correct answer. Targets the "verbatim copy"
    artifact: a hallucinated answer may share entity tokens but a correct
    answer may share long verbatim runs with the knowledge. We break those
    runs via:
      (a) Number ↔ word swap on small integers
      (b) Conservative synonym substitution
      (c) Light contraction expansion ("won't" → "will not")

    Stays semantically identical (the small synonyms are drop-ins) so the
    paraphrased answer remains grounded by the knowledge."""
    if not right.strip():
        return right
    out = right

    # (a) Number ↔ word swap, ~50% probability per token to avoid total flip.
    def _num_to_word(m: re.Match) -> str:
        n = m.group()
        if n in _NUMBER_TO_WORD and rng.random() < 0.6:
            return _NUMBER_TO_WORD[n]
        return n

    out = re.sub(r"\b\d+\b", _num_to_word, out)

    def _word_to_num(m: re.Match) -> str:
        w = m.group().lower()
        if w in _WORD_TO_NUMBER and rng.random() < 0.4:
            return _WORD_TO_NUMBER[w]
        return m.group()

    out = re.sub(
        r"\b(?:" + "|".join(_WORD_TO_NUMBER.keys()) + r")\b",
        _word_to_num,
        out,
        flags=re.IGNORECASE,
    )

    # (b) Synonym swap, first applicable per pass.
    for src, dst in _SYNONYM_SWAPS:
        if src in out.lower():
            # Case-preserving replacement at the first occurrence
            idx = out.lower().find(src)
            if idx >= 0:
                # Preserve surrounding case of first char
                head = out[:idx]
                tail = out[idx + len(src) :]
                replacement = dst
                if out[idx].isupper():
                    replacement = dst[0].upper() + dst[1:]
                out = head + replacement + tail
                break  # at most one swap to stay conservative

    # (c) Light contraction expansion
    contractions = [
        ("won't", "will not"),
        ("can't", "cannot"),
        ("isn't", "is not"),
        ("aren't", "are not"),
        ("don't", "do not"),
        ("doesn't", "does not"),
    ]
    for src, dst in contractions:
        out = out.replace(src, dst)

    return out


# ── GPT backend (reuses fusion4_falsification.py prompts) ───────────────


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


def _load_env() -> None:
    """Mirrors fusion4_falsification.py:_load_env()."""
    f = _REPO_ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(
            r"(?:export\s+)?OPENAI_API_KEY\s*=\s*"
            r"[\"']?([^\"'\s]+)",
            line.strip(),
        )
        if m:
            os.environ["OPENAI_API_KEY"] = m.group(1)


def _gpt_call(client: Any, system: str, user: str, model: str = "gpt-4o-mini") -> str:
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model,
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


# ── Dataset construction ────────────────────────────────────────────────


def build_records(
    items: list[FalsItem],
    *,
    backend: str = "deterministic",
    seed: int = 42,
    cache_path: Path | None = None,
    gpt_workers: int = 8,
    gpt_model: str = "gpt-4o-mini",
) -> list[FalsRecord]:
    """Generate the 4-condition dataset for the given items.

    When `cache_path` is provided and exists, results are loaded from it.
    Cache schema is backend-specific (the backend tag is stored in each
    record so mixed-backend caches will be detected and rebuilt).
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached_raw = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = [FalsRecord(**r) for r in cached_raw]
            if cached and cached[0].backend == backend and len(cached) == len(items):
                logger.info(
                    "loaded cache %s (n=%d backend=%s)",
                    cache_path.name,
                    len(cached),
                    backend,
                )
                return cached
            logger.info("cache backend mismatch — rebuilding")
        except Exception:  # noqa: BLE001
            logger.info("cache unreadable — rebuilding")

    rng = random.Random(seed)
    records: list[FalsRecord] = []

    if backend == "deterministic":
        for it in items:
            records.append(
                FalsRecord(
                    item_id=it.item_id,
                    context=it.context,
                    query=it.query,
                    right=it.right,
                    halu=it.halu,
                    h_ctrl=deterministic_h_ctrl(it.context, it.halu, rng=rng),
                    r_para=deterministic_r_para(it.context, it.right, rng=rng),
                    backend="deterministic",
                )
            )
    elif backend == "gpt-4o-mini":
        _load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not available for gpt-4o-mini backend")
        from openai import OpenAI

        client = OpenAI()

        def _work(idx: int, it: FalsItem) -> tuple[int, FalsRecord]:
            kq = f"Knowledge: {it.context}\n"
            if it.query:
                kq += f"Question: {it.query}\n"
            h_ctrl = _gpt_call(
                client,
                _H_CTRL_SYS,
                kq + f"Wrong answer to rewrite: {it.halu}",
                gpt_model,
            )
            r_para = _gpt_call(
                client,
                _R_PARA_SYS,
                kq + f"Correct answer to paraphrase: {it.right}",
                gpt_model,
            )
            return idx, FalsRecord(
                item_id=it.item_id,
                context=it.context,
                query=it.query,
                right=it.right,
                halu=it.halu,
                h_ctrl=h_ctrl or it.halu,
                r_para=r_para or it.right,
                backend=gpt_model,
            )

        records = [None] * len(items)  # type: ignore
        with ThreadPoolExecutor(max_workers=gpt_workers) as ex:
            futs = [ex.submit(_work, i, it) for i, it in enumerate(items)]
            done = 0
            t0 = time.perf_counter()
            for f in as_completed(futs):
                idx, rec = f.result()
                records[idx] = rec
                done += 1
                if done % 100 == 0:
                    print(
                        f"    gpt {done}/{len(items)} "
                        f"({time.perf_counter() - t0:.0f}s)",
                        flush=True,
                    )
    else:
        raise ValueError(f"unknown backend {backend!r}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
        )
        logger.info("wrote cache %s", cache_path.name)

    return records


# ── Falsification probe ─────────────────────────────────────────────────


def run_probe(
    name: str,
    records: list[FalsRecord],
    *,
    profile: str = "benchmark_qa",
    target: float | None = None,
    tolerance: float = 0.03,
) -> dict[str, Any]:
    """Score the C1-C4 conditions for the given records.

    `target` is the pre-registered AUROC target for this benchmark; the
    `survives_falsification` flag is set iff min(C1..C4) >= target - tolerance.

    Returns a canonical falsification JSON record (same shape as
    benchmarks/results/fusion4_falsification.json)."""
    from entroly.witness import WitnessAnalyzer

    analyzer = WitnessAnalyzer(use_nli=False, force_python=True, profile=profile)

    # Compute signals once per (record, variant). 4 variants × N records.
    R: list[tuple[float, float, float, float]] = []
    H: list[tuple[float, float, float, float]] = []
    HC: list[tuple[float, float, float, float]] = []
    RP: list[tuple[float, float, float, float]] = []
    gR = gH = gHC = gRP = 0.0
    t0 = time.perf_counter()
    n = len(records)
    for i, rec in enumerate(records):
        ctx = f"{rec.context}\n\nQuestion: {rec.query}" if rec.query else rec.context
        sr = _signals(analyzer, ctx, rec.right)
        sh = _signals(analyzer, ctx, rec.halu)
        shc = _signals(analyzer, ctx, rec.h_ctrl)
        srp = _signals(analyzer, ctx, rec.r_para)
        R.append(sr)
        H.append(sh)
        HC.append(shc)
        RP.append(srp)
        gR += sr[2]
        gH += sh[2]
        gHC += shc[2]
        gRP += srp[2]
        if (i + 1) % 100 == 0:
            print(
                f"    signals {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True
            )

    def _cond(neg: list, pos: list, cname: str) -> dict[str, float]:
        sc = [_fuse(x) for x in neg] + [_fuse(x) for x in pos]
        lb = [0] * len(neg) + [1] * len(pos)
        return {
            "name": cname,
            "fusion": auroc(sc, lb),
            "g_only": auroc([x[2] for x in neg] + [x[2] for x in pos], lb),
            "witness_only": auroc([x[0] for x in neg] + [x[0] for x in pos], lb),
        }

    C1 = _cond(R, H, "C1 original (r vs h)")
    C2 = _cond(R, HC, "C2 entity-controlled (r vs h_ctrl)")
    C3 = _cond(RP, H, "C3 paraphrase-stress (r_para vs h)")
    C4 = _cond(RP, HC, "C4 realistic (r_para vs h_ctrl)")

    aurocs = [c["fusion"] for c in (C1, C2, C3, C4)]
    min_auc = min(aurocs)
    artifact_drop = C1["fusion"] - C4["fusion"]
    # Same falsification rules as fusion4_falsification.py:262-266
    hc_collapse = abs((gHC - gR) / n) <= 0.05 if n else False
    rp_inflate = abs((gRP - gH) / n) <= 0.05 if n else False
    artifact_detected = (artifact_drop >= 0.07) or hc_collapse or rp_inflate

    survives_threshold: float | None
    survives_falsification: bool | None
    if target is None:
        survives_threshold = None
        survives_falsification = None
    else:
        survives_threshold = target - tolerance
        survives_falsification = min_auc >= survives_threshold

    return {
        "benchmark": name,
        "n": n,
        "weights": {"W": WEIGHTS[0], "E": WEIGHTS[1], "G": WEIGHTS[2], "S": WEIGHTS[3]},
        "backend": records[0].backend if records else "unknown",
        "conditions": [C1, C2, C3, C4],
        "min_fusion_auroc_c1_c4": min_auc,
        "max_fusion_auroc_c1_c4": max(aurocs),
        "artifact_drop_c1_c4": artifact_drop,
        "mean_g": {
            "r": gR / n if n else 0.0,
            "h": gH / n if n else 0.0,
            "h_ctrl": gHC / n if n else 0.0,
            "r_para": gRP / n if n else 0.0,
        },
        "artifact_detected": artifact_detected,
        "target": target,
        "survives_threshold": survives_threshold,
        "survives_falsification": survives_falsification,
    }


def dataset_hash(items: list[FalsItem]) -> str:
    """SHA256 over a stable serialization of the dataset, first 16 hex chars."""
    h = hashlib.sha256()
    for it in items:
        line = f"{it.item_id}|{it.context[:200]}|{it.query[:100]}|{it.right[:100]}|{it.halu[:100]}\n".encode()
        h.update(line)
    return h.hexdigest()[:16]
