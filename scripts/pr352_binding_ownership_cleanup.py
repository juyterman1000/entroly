from __future__ import annotations

from pathlib import Path


CORE = Path("entroly-core/src")


def move_binding(old_name: str, new_name: str) -> None:
    old = CORE / old_name
    new = CORE / new_name
    if not old.is_file():
        raise SystemExit(f"expected binding source missing: {old}")
    if new.exists():
        raise SystemExit(f"destination already exists: {new}")
    new.write_bytes(old.read_bytes())
    old.unlink()


move_binding("cognitive_bus.rs", "cognitive_bus_bindings.rs")
move_binding("nkbe.rs", "nkbe_bindings.rs")

lib = CORE / "lib.rs"
text = lib.read_text(encoding="utf-8")
replacements = (
    (
        "mod cognitive_bus;",
        '#[path = "cognitive_bus_bindings.rs"]\nmod cognitive_bus;',
    ),
    (
        "mod nkbe;",
        '#[path = "nkbe_bindings.rs"]\nmod nkbe;',
    ),
)
for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"lib.rs module anchor changed: {old!r}: {count}")
    text = text.replace(old, new, 1)
lib.write_text(text, encoding="utf-8")

# Coordination was promoted after its falsification harness proved equivalence.
# Keep the source documentation aligned with its actual production ownership.
coord = Path("entroly-engine/src/coordination_index.rs")
text = coord.read_text(encoding="utf-8")
old_header = '''//! Falsification harness for scalable Work Graph lease candidate generation.
//!
//! This module is test-only until the indexed candidate generator has proven
//! semantic equivalence to the current all-pairs implementation. The exact
//! overlap functions in `work_graph` remain authoritative in production.
'''
new_header = '''//! Scalable Work Graph lease candidate generation.
//!
//! The indexed generator is production-wired only as a candidate filter after
//! randomized equivalence testing against the naive all-pairs oracle. The exact
//! overlap functions in `work_graph` remain authoritative for conflict semantics.
'''
if text.count(old_header) != 1:
    raise SystemExit("coordination module documentation anchor changed")
text = text.replace(old_header, new_header, 1)
old_doc = "/// decision after this candidate filter is promoted."
new_doc = "/// decision after this candidate filter."
if text.count(old_doc) != 1:
    raise SystemExit("coordination candidate documentation anchor changed")
coord.write_text(text.replace(old_doc, new_doc, 1), encoding="utf-8")

print("binding ownership cleanup applied")
