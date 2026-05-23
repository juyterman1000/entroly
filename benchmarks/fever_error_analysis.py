"""
FEVER error analysis. Shows the worst false negatives and false positives
so we can see WHY we're stuck at 0.768 and what's missing.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from entroly.esg import ESGAnalyzer

SEED = 42
N_SAMPLES = 1000


def load_fever():
    from datasets import load_dataset
    ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
    label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": None,
                 "NOT_ENOUGH_INFO": None}
    out = []
    for row in ds:
        claim = str(row.get("claim", "") or "")
        ev_field = row.get("evidence") or ""
        if isinstance(ev_field, list):
            parts = []
            for x in ev_field:
                if isinstance(x, list):
                    sent = ""
                    for el in x:
                        if isinstance(el, str) and len(el) > len(sent):
                            sent = el
                    if sent:
                        parts.append(sent)
                elif isinstance(x, dict):
                    parts.append(str(x.get("sentence", x.get("text", ""))))
                else:
                    parts.append(str(x))
            ev_text = "\n".join(p for p in parts if p)
        else:
            ev_text = str(ev_field)
        label = label_map.get(str(row.get("label", "") or "").upper().strip())
        if claim and ev_text and label is not None:
            out.append((ev_text, claim, label))
    return out


def main() -> int:
    print("Loading FEVER...")
    items = load_fever()
    rng = random.Random(SEED)
    rng.shuffle(items)
    binary = items[:N_SAMPLES]

    esg = ESGAnalyzer()

    # Score all and find threshold
    scored = []
    for ev, cl, lab in binary:
        t = esg.tension(ev, cl)
        scored.append((t, lab, cl, ev))

    # Sort by score (T(G))
    scored.sort()

    # Pick balanced-accuracy threshold
    n_pos = sum(1 for _, lab, _, _ in scored if lab == 1)
    n_neg = sum(1 for _, lab, _, _ in scored if lab == 0)
    print(f"\nN positives (REFUTES): {n_pos}  N negatives (SUPPORTS): {n_neg}")

    # Best-acc threshold
    best_acc = 0
    best_t = 0.5
    tp = fp = 0
    for s, y, _, _ in reversed(scored):
        if y == 1: tp += 1
        else: fp += 1
        tn = n_neg - fp; fn = n_pos - tp
        acc = (tp + tn) / (n_pos + n_neg)
        if acc > best_acc:
            best_acc = acc
            best_t = s
    print(f"Best threshold = {best_t:.4f}, acc = {best_acc:.4f}")
    print()

    # FALSE NEGATIVES: REFUTES with LOW T(G) — we said supported but it's a hallucination
    print("=" * 80)
    print("WORST 15 FALSE NEGATIVES (REFUTES we missed — low T(G), said SUPPORTS)")
    print("=" * 80)
    fns = [(s, lab, cl, ev) for s, lab, cl, ev in scored if lab == 1 and s < best_t]
    fns.sort()   # sort low to high T(G)
    for s, lab, cl, ev in fns[:15]:
        print(f"\n  T(G)={s:.4f}")
        print(f"  CLAIM: {cl}")
        print(f"  EV   : {ev[:200]}{'...' if len(ev) > 200 else ''}")

    print()
    print("=" * 80)
    print("WORST 15 FALSE POSITIVES (SUPPORTS we wrongly flagged — high T(G), said REFUTES)")
    print("=" * 80)
    fps = [(s, lab, cl, ev) for s, lab, cl, ev in scored if lab == 0 and s >= best_t]
    fps.sort(reverse=True)
    for s, lab, cl, ev in fps[:15]:
        print(f"\n  T(G)={s:.4f}")
        print(f"  CLAIM: {cl}")
        print(f"  EV   : {ev[:200]}{'...' if len(ev) > 200 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
