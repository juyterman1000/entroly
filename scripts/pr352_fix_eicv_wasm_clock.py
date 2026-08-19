#!/usr/bin/env python3
"""Make EICV latency telemetry safe on wasm32 without changing trust semantics.

`std::time::Instant::now()` compiles for `wasm32-unknown-unknown` but traps at
runtime. EICV latency is telemetry only, so native builds keep monotonic timing
while wasm emits a deterministic 0.0 ms. Scoring, decisions, evidence hashes,
and Trust Engine commitments are unchanged.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "entroly-engine/src/eicv.rs"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    marker = '#[cfg(not(target_arch = "wasm32"))]\n        let start = std::time::Instant::now();'
    if marker in text:
        print("EICV wasm-safe timing already applied")
        return 0

    text = replace_once(
        text,
        "        let start = std::time::Instant::now();\n",
        '        #[cfg(not(target_arch = "wasm32"))]\n'
        "        let start = std::time::Instant::now();\n",
        "EICV timing start",
    )
    text = replace_once(
        text,
        "        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;\n",
        '        #[cfg(not(target_arch = "wasm32"))]\n'
        "        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;\n"
        '        #[cfg(target_arch = "wasm32")]\n'
        "        let elapsed_ms = 0.0;\n",
        "EICV elapsed telemetry",
    )
    PATH.write_text(text, encoding="utf-8")
    print("applied wasm-safe EICV timing telemetry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
