"""Apply the PR #280 WASM instance-counter safety repair deterministically."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "entroly-wasm" / "src" / "lib.rs"
    text = path.read_text(encoding="utf-8")

    old_import = "use std::collections::{HashMap, HashSet};\n"
    new_import = (
        "use std::collections::{HashMap, HashSet};\n"
        "use std::sync::atomic::{AtomicU32, Ordering};\n"
    )
    if new_import not in text:
        if text.count(old_import) != 1:
            raise RuntimeError("expected exactly one collections import")
        text = text.replace(old_import, new_import, 1)

    old_counter = '''        // Generate instance_id (simple counter for wasm — no multi-threading)
        static mut INSTANCE_COUNTER: u64 = 0;
        let instance_id = unsafe {
            INSTANCE_COUNTER += 1;
            let raw = INSTANCE_COUNTER;
            let mut x = raw.wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        };
'''
    new_counter = '''        // Use an atomic even though the default wasm target is single-threaded.
        // This removes unnecessary `static mut` undefined-behavior risk under
        // re-entrant construction and keeps the code sound for threaded wasm.
        static INSTANCE_COUNTER: AtomicU32 = AtomicU32::new(0);
        let raw = u64::from(
            INSTANCE_COUNTER
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_add(1),
        );
        let mut x = raw.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        let instance_id = x ^ (x >> 31);
'''
    if new_counter not in text:
        if text.count(old_counter) != 1:
            raise RuntimeError(
                f"expected exactly one unsafe instance counter, found {text.count(old_counter)}"
            )
        text = text.replace(old_counter, new_counter, 1)

    test_marker = "fn wasm_engine_instances_receive_distinct_ids()"
    if test_marker not in text:
        text = text.rstrip() + '''


#[cfg(test)]
mod instance_counter_tests {
    use super::*;

    #[test]
    fn wasm_engine_instances_receive_distinct_ids() {
        let first = WasmEntrolyEngine::new();
        let second = WasmEntrolyEngine::new();
        assert_ne!(first.instance_id, second.instance_id);
        assert_ne!(first.rng_state, second.rng_state);
    }
}
'''

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
