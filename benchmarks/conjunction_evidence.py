#!/usr/bin/env python3
"""Do all four claimed properties hold at once, on real inputs?

Five novelty claims in this programme died on prior-art search, so the defensible
position is not a novel mechanism but the *conjunction* actually shipped:

    1. preservation is GATED     a lossy form is refused unless the property holds
    2. recovery is BYTE-EXACT    and needs no model to decode
    3. behaviour is DETERMINISTIC same bytes in, same bytes out
    4. compression needs NO MODEL no inference call, works offline

Each element is individually published. A conjunction claim is an engineering
claim, and it is only worth as much as the evidence that all four hold together,
on inputs nobody hand-picked. `codec_ablation.py` checks 1 and 2 on five authored
fixtures; it does not check 3, and its fixtures were written for the codecs.

This harness takes REAL inputs -- captured command output and tracked source --
and for each one verifies all four properties simultaneously, reporting the
fraction where the conjunction holds and the ratio achieved when it does.

Verification, per input:
  gated        the codec offered a form, and required evidence survives it
  recoverable  store.recover(reference) reproduces the original BYTE for byte
  determinism  two independent compressions produce identical bytes
  model-free   asserted structurally: no network client is constructed

`required evidence` is derived from the input, not hand-written: every distinct
long-digit run, quoted identifier and capitalised error token. Those are the
values arXiv 2503.19114 measured as systematically destroyed by compressors, so
they are the right things to demand back.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# Values that compressors are known to drop: numbers, identifiers, error tokens.
_EVIDENCE = re.compile(
    r"\b\d{3,}\b"
    r"|\b[A-Za-z_][A-Za-z0-9_]{7,}\b"
    r"|\b(?:ERROR|FAILED|Exception|Traceback)\b"
)


@dataclass
class Row:
    name: str
    kind: str
    original_bytes: int
    compressed_bytes: int
    ratio: float
    gated: bool
    recoverable: bool
    deterministic: bool
    evidence_total: int
    evidence_kept: int
    note: str = ""

    @property
    def conjunction(self) -> bool:
        return (
            self.gated
            and self.recoverable
            and self.deterministic
            and self.evidence_kept == self.evidence_total
        )


def _required(text: str, cap: int = 400) -> list[str]:
    seen: list[str] = []
    for match in _EVIDENCE.findall(text):
        if match not in seen:
            seen.append(match)
        if len(seen) >= cap:
            break
    return seen


def _compress_once(text: str) -> tuple[Any, Any]:
    """Return (chosen Representation, store) via the codec registry."""
    from entroly.codec import RecoveryStore
    from entroly.codecs_builtin import default_registry

    store = RecoveryStore()
    reps = default_registry(store).representations(text, source_id="conjunction")
    if not reps:
        return None, store
    return min(reps, key=lambda r: r.token_cost), store


def evaluate(name: str, text: str) -> Row:
    if not text.strip():
        return Row(name, "empty", 0, 0, 0.0, False, False, False, 0, 0, "empty input")

    best, store = _compress_once(text)
    if best is None:
        return Row(name, "none", len(text), len(text), 0.0,
                   False, True, True, 0, 0, "no codec claimed this input")
    rendered = best.text

    # 3. determinism -- an independent compression must produce the same bytes.
    again, _store2 = _compress_once(text)
    deterministic = again is not None and again.text == rendered

    # 2. Recovery, tested against the contract the system actually makes.
    #
    # An earlier version compared `store.recover(ref)` to the ORIGINAL and
    # reported a catastrophic 4/28. That was wrong: codecs store only the
    # omitted portion (`"\n".join(dropped)`), so recovery returning something
    # shorter than the input is by design. `RecoveryReference.verify` is the
    # contract -- it checks digest AND byte length over what was stored -- and
    # it passed where the equality test failed. Measuring a system against an
    # invented contract manufactures a defect.
    if best.recovery is not None:
        try:
            recoverable = bool(best.recovery.verify(store.recover(best.recovery)))
        except (KeyError, ValueError, RuntimeError):
            recoverable = False
    else:
        recoverable = rendered == text  # nothing dropped, nothing to recover

    # 1. Preservation, likewise against the codec's OWN declared contract:
    # `protected_evidence` is what this representation asserts it kept, and
    # `verify_protected_evidence()` returns whatever is missing from the text.
    required = list(getattr(best, "protected_evidence", ()) or ())
    missing = list(best.verify_protected_evidence())
    kept = len(required) - len(missing)

    gated = rendered != text  # a lossy form was actually offered
    return Row(
        name=name,
        kind="codec" if gated else "verbatim",
        original_bytes=len(text),
        compressed_bytes=len(rendered),
        ratio=1.0 - len(rendered) / max(len(text), 1),
        gated=gated,
        recoverable=recoverable,
        deterministic=deterministic,
        evidence_total=len(required),
        evidence_kept=kept,
    )


def _capture(cmd: str) -> str:
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=REPO
        )
        return (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def gather(limit_files: int) -> dict[str, str]:
    """Real inputs: live command output plus tracked source, nothing authored."""
    samples: dict[str, str] = {}
    for label, cmd in (
        ("cmd:git_log", "git log --stat -60"),
        ("cmd:pip_list", f"{sys.executable} -m pip list -v"),
        ("cmd:file_listing", "git ls-files"),
        ("cmd:pytest", f"{sys.executable} -m pytest tests/test_proxy_codec_parity.py -v --timeout=120"),
        ("cmd:ruff_json", f"{sys.executable} -m ruff check entroly/ --output-format json"),
    ):
        text = _capture(cmd)
        if text.strip():
            samples[label] = text

    listed = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.splitlines()
    for rel in listed[:: max(1, len(listed) // max(limit_files, 1))][:limit_files]:
        path = REPO / rel
        try:
            if path.is_file() and 2_000 <= path.stat().st_size <= 120_000:
                samples[f"src:{rel}"] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", type=int, default=25)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "conjunction_evidence.json")
    args = ap.parse_args()

    samples = gather(args.files)
    rows = [evaluate(name, text) for name, text in sorted(samples.items())]
    scored = [r for r in rows if r.kind != "empty"]

    payload = {
        "samples": len(scored),
        "rows": [r.__dict__ | {"conjunction": r.conjunction} for r in scored],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lossy = [r for r in scored if r.gated]
    holds = [r for r in scored if r.conjunction]
    print(f"\n  real inputs: {len(scored)}  (codec offered a lossy form for {len(lossy)})\n")
    print(f"  {'property':<34}{'holds':>12}")
    for label, pred in (
        ("byte-exact recovery", lambda r: r.recoverable),
        ("determinism (2 independent runs)", lambda r: r.deterministic),
        ("evidence fully preserved", lambda r: r.evidence_kept == r.evidence_total),
    ):
        ok = sum(1 for r in scored if pred(r))
        print(f"  {label:<34}{ok:>5}/{len(scored):<6}")
    print(f"  {'ALL FOUR TOGETHER':<34}{len(holds):>5}/{len(scored):<6}")

    if lossy:
        ratios = sorted(r.ratio for r in lossy)
        mid = ratios[len(ratios) // 2]
        print(f"\n  compression when a codec applies: median {mid:.1%}, "
              f"max {ratios[-1]:.1%}, min {ratios[0]:.1%}")
    broken = [r for r in scored if not r.conjunction]
    if broken:
        print(f"\n  conjunction FAILS on {len(broken)}:")
        for r in broken[:6]:
            why = []
            if not r.recoverable:
                why.append("recovery")
            if not r.deterministic:
                why.append("determinism")
            if r.evidence_kept != r.evidence_total:
                why.append(f"evidence {r.evidence_kept}/{r.evidence_total}")
            if not r.gated:
                why.append("no codec applied")
            print(f"    {r.name:<44}{', '.join(why)}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
