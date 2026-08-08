#!/usr/bin/env python3
"""Re-read cost: content-digest caching against a time-bounded external cache.

An agent re-reads the same files constantly. The external context runtime
handles this with a read cache and it works well -- a warm re-read measured
2,171 tokens down to 8. It is time-bounded, though: in the same session, a file
re-read later returned full content again because the entry had aged out.

`entroly.session_read_cache` keys on the content digest instead. The claim under
test is that this is better on both axes that matter:

  durability   an unchanged file stays free however long the session runs
  correctness  a modified file is ALWAYS delivered in full, because its digest
               no longer matches -- a stale reference cannot be constructed

The session simulated here is the realistic one: read a working set, do other
work, then come back to the same files. Both systems see identical files in an
identical order.

The external arm is invoked through `ENTROLY_EXTERNAL_CTX_BIN`; its product name
is deliberately absent from this source per the external-name policy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _external_read(binary: str, rel: str) -> int:
    try:
        proc = subprocess.run([binary, "read", rel, "-m", "full"], cwd=REPO,
                              capture_output=True, text=True, timeout=180,
                              encoding="utf-8", errors="replace")
        return _tokens(proc.stdout or "")
    except (OSError, subprocess.SubprocessError):
        return 0


def run(files: list[str], binary: str | None, revisits: int) -> dict[str, Any]:
    from entroly.session_read_cache import SessionReadCache

    texts = {rel: (REPO / rel).read_text(encoding="utf-8", errors="replace") for rel in files}
    cache = SessionReadCache()

    entroly_cold = entroly_warm = 0
    external_cold = external_warm = 0

    # Pass 1 -- the working set is read for the first time.
    for rel in files:
        entroly_cold += cache.deliver(rel, texts[rel]).delivered_tokens
        if binary:
            external_cold += _external_read(binary, rel)

    # Interleaved work, then revisits -- the pattern that exposes a TTL.
    for _ in range(revisits):
        cache.advance_turn()
        for rel in files:
            entroly_warm += cache.deliver(rel, texts[rel]).delivered_tokens
            if binary:
                external_warm += _external_read(binary, rel)

    raw_total = sum(_tokens(t) for t in texts.values())

    # Correctness probe: a modified file must never be suppressed.
    probe_rel = files[0]
    cache.advance_turn()
    modified = texts[probe_rel] + "\n\ndef appended_after_caching(x):\n    return x\n"
    probe = cache.deliver(probe_rel, modified)

    return {
        "files": len(files),
        "revisits": revisits,
        "raw_tokens_per_pass": raw_total,
        "entroly": {
            "cold_tokens": entroly_cold,
            "warm_tokens_total": entroly_warm,
            "warm_tokens_per_pass": entroly_warm // max(revisits, 1),
            "stats": cache.stats(),
        },
        "external": {
            "cold_tokens": external_cold,
            "warm_tokens_total": external_warm,
            "warm_tokens_per_pass": external_warm // max(revisits, 1),
        } if binary else None,
        "modified_file_probe": {
            "suppressed": probe.suppressed,
            "delivered_tokens": probe.delivered_tokens,
            "expected": "not suppressed; a changed file must be delivered in full",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--revisits", type=int, default=3)
    ap.add_argument("--pattern", default="entroly/*.py")
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "read_cache_comparison.json")
    args = ap.parse_args()

    binary = os.environ.get("ENTROLY_EXTERNAL_CTX_BIN")
    if binary and not Path(binary).exists():
        binary = None

    listed = subprocess.run(["git", "ls-files", args.pattern], cwd=REPO,
                            capture_output=True, text=True, check=False).stdout.splitlines()
    picked = [rel for rel in listed
              if (REPO / rel).is_file() and 6_000 <= (REPO / rel).stat().st_size <= 60_000][:args.limit]
    if not picked:
        print("no files matched")
        return 1

    payload = run(picked, binary, args.revisits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    raw = payload["raw_tokens_per_pass"]
    ent = payload["entroly"]
    print(f"\n  {payload['files']} files, {payload['revisits']} revisit passes")
    print(f"  raw cost of one pass: {raw:,} tokens\n")
    print(f"  {'system':<12}{'cold pass':>12}{'warm pass':>12}{'warm vs raw':>14}")
    print(f"  {'entroly':<12}{ent['cold_tokens']:>12,}{ent['warm_tokens_per_pass']:>12,}"
          f"{1 - ent['warm_tokens_per_pass'] / max(raw, 1):>13.2%}")
    ext = payload["external"]
    if ext:
        print(f"  {'external':<12}{ext['cold_tokens']:>12,}{ext['warm_tokens_per_pass']:>12,}"
              f"{1 - ext['warm_tokens_per_pass'] / max(raw, 1):>13.2%}")
    probe = payload["modified_file_probe"]
    verdict = "PASS" if not probe["suppressed"] else "FAIL"
    print(f"\n  modified-file probe: {verdict} "
          f"({'delivered in full' if not probe['suppressed'] else 'WRONGLY SUPPRESSED'})")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
