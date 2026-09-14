"""Tests for Spec 3: Speculative Assembly — pre-assemble predicted context."""

from __future__ import annotations

from pathlib import Path

from entroly.prefetch import (
    AssembledFragment,
    PrefetchEngine,
    SpeculativeAssembler,
)


def _write(root: Path, name: str, content: str) -> None:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_assembler_loads_predicted_imports(tmp_path: Path) -> None:
    _write(tmp_path, "utils.py", "def helper():\n    return 1\n")
    _write(tmp_path, "main.py", "from utils import helper\n\nhelper()\n")

    engine = PrefetchEngine()
    assembler = SpeculativeAssembler(
        str(tmp_path), engine, max_assembly_tokens=8192
    )

    fragments = assembler.assemble_for(
        "main.py",
        "from utils import helper\n\nhelper()\n",
    )

    loaded_paths = {f.path for f in fragments}
    assert any("utils" in p for p in loaded_paths)


def test_assembler_respects_token_budget(tmp_path: Path) -> None:
    _write(tmp_path, "big.py", "x = 1\n" * 5000)
    _write(tmp_path, "main.py", "from big import x\n")

    engine = PrefetchEngine()
    assembler = SpeculativeAssembler(
        str(tmp_path), engine, max_assembly_tokens=100
    )

    fragments = assembler.assemble_for(
        "main.py",
        "from big import x\n",
    )

    total_tokens = sum(f.token_estimate for f in fragments)
    assert total_tokens <= 100


def test_assembler_skips_nonexistent_files(tmp_path: Path) -> None:
    _write(tmp_path, "app.py", "from nonexistent import foo\n")

    engine = PrefetchEngine()
    assembler = SpeculativeAssembler(str(tmp_path), engine)

    fragments = assembler.assemble_for(
        "app.py",
        "from nonexistent import foo\n",
    )

    for f in fragments:
        assert f.content


def test_assembler_caches_and_returns(tmp_path: Path) -> None:
    _write(tmp_path, "lib.py", "def lib_func():\n    pass\n")
    _write(tmp_path, "caller.py", "from lib import lib_func\n")

    engine = PrefetchEngine()
    assembler = SpeculativeAssembler(str(tmp_path), engine)

    assembler.assemble_for("caller.py", "from lib import lib_func\n")
    cached = assembler.get_cached("caller.py")

    assert cached is not None
    assert assembler.get_cached("caller.py") is None


def test_assembler_filters_low_confidence(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", "x = 1\n")

    engine = PrefetchEngine()
    assembler = SpeculativeAssembler(
        str(tmp_path), engine, min_confidence=0.99
    )

    fragments = assembler.assemble_for("mod.py", "x = 1\n")
    assert len(fragments) == 0


def test_assembler_stats(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "import os\n")

    engine = PrefetchEngine()
    assembler = SpeculativeAssembler(str(tmp_path), engine)

    assembler.assemble_for("a.py", "import os\n")
    stats = assembler.stats()

    assert stats["assemblies"] == 1
    assert stats["cache_hits"] == 0
    assert stats["pending"] == 1


def test_assembled_fragment_dataclass() -> None:
    frag = AssembledFragment(
        path="test.py",
        content="def test(): pass",
        reason="import",
        confidence=0.7,
        token_estimate=4,
    )
    assert frag.path == "test.py"
    assert frag.token_estimate == 4


def test_assembler_prevents_path_traversal(tmp_path: Path) -> None:
    _write(tmp_path, "safe.py", "x = 1\n")

    engine = PrefetchEngine()
    assembler = SpeculativeAssembler(str(tmp_path), engine)

    content = assembler._load_file("../../etc/passwd")
    assert content is None
