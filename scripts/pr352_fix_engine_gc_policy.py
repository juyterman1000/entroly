#!/usr/bin/env python3
"""Remove all process-global Python GC mutation from EntrolyEngine.

Python's cyclic GC policy is process-wide. A library must not freeze, disable,
enable, or force collection for the embedding application. This guarded repair
keeps Entroly's request semantics unchanged while removing constructor,
turn-loop, ingest, and optimize-path GC ownership.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "entroly/engine.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def dedent_four(text: str) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(line[4:] if line.startswith("    ") else line for line in lines)


def main() -> int:
    text = ENGINE.read_text(encoding="utf-8")
    if "Host GC policy is process-wide" in text:
        forbidden = ("gc.disable(", "gc.enable(", "gc.freeze(", "gc.collect(")
        remaining = [token for token in forbidden if token in text]
        if remaining:
            raise SystemExit(f"partial GC hardening detected; remaining calls: {remaining}")
        print("host GC policy already hardened")
        return 0

    text = replace_once(text, "import gc\n", "", "engine gc import")
    text = replace_once(
        text,
        """        # GC freeze at startup: Python's cyclic GC causes ~500ms stalls on large
        # heaps. Freeze all existing long-lived objects and disable automatic
        # collection. We manually collect every N tool calls in advance_turn()
        # to reclaim short-lived garbage without unpredictable pauses.
        self._gc_collect_interval = 50  # collect every 50 turns
        gc.collect()
        gc.freeze()
        gc.disable()
""",
        """        # Host GC policy is process-wide. Entroly deliberately leaves it
        # untouched: a library must not freeze, disable, enable, or collect
        # objects owned by the embedding application or its other threads.
""",
        "engine constructor GC policy",
    )
    text = replace_once(
        text,
        """        # Periodic GC amortization: frozen at init, collect every N turns
        if self._turn_counter > 0 and self._turn_counter % self._gc_collect_interval == 0:
            gc.collect()

""",
        "",
        "engine periodic GC",
    )

    ingest_start = text.find("    def ingest_fragment(\n")
    ingest_end = text.find("    def remove_sources(", ingest_start)
    if ingest_start < 0 or ingest_end < 0:
        raise SystemExit("ingest_fragment boundaries not found")
    ingest = text[ingest_start:ingest_end]
    if "gc.disable()" not in ingest or "gc.enable()" not in ingest or "gc.collect()" not in ingest:
        raise SystemExit("ingest_fragment GC ownership anchors missing")
    replacement = '''    def ingest_fragment(
        self,
        content: str,
        source: str = "",
        token_count: int = 0,
        is_pinned: bool = False,
    ) -> dict[str, Any]:
        """Ingest a new context fragment without changing host GC policy."""
        # Lazy warm-start MUST run before the first mutation: load_index
        # replaces the fragment set, so it has to happen before any ingest or
        # it would wipe freshly-ingested fragments.
        self._ensure_index_loaded()

        # Invalidate the fast-path's fragment cache: the Rust engine has
        # gained a new fragment, so any cached export_fragments() snapshot
        # is now stale.
        self._fragment_cache_dirty = True
        if self._use_rust:
            # Enforce max_fragments cap on Rust engine (Rust doesn't enforce it)
            if self._rust.fragment_count() >= self.config.max_fragments:
                return {
                    "status": "rejected",
                    "reason": "max_fragments cap reached",
                    "max_fragments": self.config.max_fragments,
                }
            result = self._rust.ingest(content, source, token_count, is_pinned)
            if source:
                self._prefetch.record_access(source, self._rust.get_turn())
            if self._checkpoint_mgr.should_auto_checkpoint():
                self._auto_checkpoint()
            return dict(result)
        return self._ingest_python(content, source, token_count, is_pinned)

'''
    text = text[:ingest_start] + replacement + text[ingest_end:]

    optimize_marker = """        # GC freeze: disable during hot Rust dispatch + final result assembly.
        gc.disable()
        try:
"""
    optimize_start = text.find(optimize_marker)
    if optimize_start < 0:
        raise SystemExit("optimize-path GC start anchor missing")
    optimize_finally = """        finally:
            gc.enable()
            gc.collect()

"""
    optimize_end = text.find(optimize_finally, optimize_start)
    if optimize_end < 0:
        raise SystemExit("optimize-path GC finally anchor missing")
    body_start = optimize_start + len(optimize_marker)
    body = text[body_start:optimize_end]
    body = dedent_four(body)
    replacement = (
        "        # Keep optimization inside Entroly's own allocation discipline;\n"
        "        # never mutate the embedding process's GC policy.\n"
        + body
    )
    text = text[:optimize_start] + replacement + text[optimize_end + len(optimize_finally):]

    forbidden = ("gc.disable(", "gc.enable(", "gc.freeze(", "gc.collect(")
    remaining = [token for token in forbidden if token in text]
    if remaining:
        raise SystemExit(f"engine.py still mutates host GC policy: {remaining}")
    if "import gc" in text:
        raise SystemExit("engine.py still imports gc after GC policy removal")

    ENGINE.write_text(text, encoding="utf-8")
    print("removed all process-global GC ownership from EntrolyEngine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
