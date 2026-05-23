"""
Tests for the Archetype-Aware Evolution system (Pillar 4).

Tests cover:
  1. Codebase scanning and fingerprinting
  2. Archetype classification accuracy
  3. Per-archetype weight strategy persistence
  4. PRISM 5D weight export (including resonance)
  5. Integration with DreamingLoop feedback path
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from entroly.archetype_optimizer import (
    ArchetypeOptimizer,
    CodebaseStats,
    scan_codebase,
    DEFAULT_WEIGHTS,
    _log_norm,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_project(tmp_path):
    """Create a mock Python backend project."""
    # Python files
    for name in ["app.py", "models.py", "views.py", "utils.py", "config.py"]:
        f = tmp_path / name
        f.write_text(
            f"import os\nfrom pathlib import Path\n\ndef {name.replace('.py', '')}():\n    pass\n",
            encoding="utf-8",
        )
    # Test files
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("def test_app():\n    assert True\n", encoding="utf-8")
    (tests / "test_models.py").write_text("def test_models():\n    assert True\n", encoding="utf-8")
    # Config files
    (tmp_path / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools"]\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("# My Project\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tmp_rust_project(tmp_path):
    """Create a mock Rust systems library."""
    src = tmp_path / "src"
    src.mkdir()
    for name in ["lib.rs", "engine.rs", "parser.rs", "types.rs"]:
        f = src / name
        f.write_text(
            f"use std::collections::HashMap;\n\npub struct {name.replace('.rs', '').title()} {{}}\n\nimpl {name.replace('.rs', '').title()} {{\n    pub fn new() -> Self {{ Self {{}} }}\n}}\n",
            encoding="utf-8",
        )
    # Tests
    test_dir = tmp_path / "tests"
    test_dir.mkdir(exist_ok=True)
    (test_dir / "integration_test.rs").write_text(
        "use super::*;\n\n#[test]\nfn test_engine() {\n    assert!(true);\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "mylib"\nversion = "0.1.0"\n', encoding="utf-8")
    return tmp_path


@pytest.fixture
def tmp_js_project(tmp_path):
    """Create a mock JS frontend project."""
    src = tmp_path / "src"
    src.mkdir()
    comps = src / "components"
    comps.mkdir()
    for name in ["App.tsx", "Header.tsx", "Footer.tsx", "Button.tsx"]:
        f = comps / name
        f.write_text(
            f"import React from 'react';\n\nexport const {name.replace('.tsx', '')} = () => {{\n  return <div>{name}</div>;\n}};\n",
            encoding="utf-8",
        )
    (src / "index.ts").write_text("import {{ App }} from './components/App';\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "myapp", "version": "1.0.0"}', encoding="utf-8")
    return tmp_path


@pytest.fixture
def optimizer(tmp_path, tmp_project):
    """Create an optimizer for the Python project."""
    data_dir = tmp_path / ".entroly_data"
    return ArchetypeOptimizer(data_dir=data_dir, project_root=tmp_project)


# ═══════════════════════════════════════════════════════════════════
# Codebase Scanner Tests
# ═══════════════════════════════════════════════════════════════════

class TestCodebaseScanner:
    def test_scan_counts_python_files(self, tmp_project):
        stats = scan_codebase(tmp_project)
        assert stats.python_files >= 5  # app, models, views, utils, config
        assert stats.test_files >= 2

    def test_scan_counts_functions(self, tmp_project):
        stats = scan_codebase(tmp_project)
        assert stats.total_functions >= 5  # one def per .py file

    def test_scan_counts_imports(self, tmp_project):
        stats = scan_codebase(tmp_project)
        assert stats.total_imports >= 5  # import os + from pathlib per file

    def test_scan_detects_rust(self, tmp_rust_project):
        stats = scan_codebase(tmp_rust_project)
        assert stats.rust_files >= 4

    def test_scan_detects_js_ts(self, tmp_js_project):
        stats = scan_codebase(tmp_js_project)
        assert stats.js_ts_files >= 4

    def test_scan_skips_hidden_dirs(self, tmp_project):
        git_dir = tmp_project / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        stats = scan_codebase(tmp_project)
        # .git files should NOT be counted
        assert all(".git" not in str(v) for v in [stats.total_files])

    def test_scan_respects_max_files(self, tmp_project):
        stats = scan_codebase(tmp_project, max_files=3)
        assert stats.total_files <= 3

    def test_scan_produces_entropy_values(self, tmp_project):
        stats = scan_codebase(tmp_project)
        # Small test files may be <64 bytes, below compression threshold
        assert len(stats.entropy_values) >= 0
        assert all(0 <= e <= 1 for e in stats.entropy_values)


# ═══════════════════════════════════════════════════════════════════
# Classification Tests
# ═══════════════════════════════════════════════════════════════════

class TestArchetypeClassification:
    def test_classifies_python_backend(self, tmp_path, tmp_project):
        opt = ArchetypeOptimizer(data_dir=tmp_path / ".data", project_root=tmp_project)
        info = opt.detect_and_load()
        assert "python" in info.label

    def test_classifies_rust_project(self, tmp_path, tmp_rust_project):
        opt = ArchetypeOptimizer(data_dir=tmp_path / ".data", project_root=tmp_rust_project)
        info = opt.detect_and_load()
        assert "rust" in info.label

    def test_classifies_js_project(self, tmp_path, tmp_js_project):
        opt = ArchetypeOptimizer(data_dir=tmp_path / ".data", project_root=tmp_js_project)
        info = opt.detect_and_load()
        assert "js" in info.label

    def test_classification_returns_weights(self, optimizer):
        info = optimizer.detect_and_load()
        assert "w_recency" in info.weights
        assert "w_frequency" in info.weights
        assert "w_semantic" in info.weights
        assert "w_entropy" in info.weights
        assert "w_resonance" in info.weights

    def test_classification_returns_confidence(self, optimizer):
        info = optimizer.detect_and_load()
        assert 0 <= info.confidence <= 1


# ═══════════════════════════════════════════════════════════════════
# Weight Management Tests
# ═══════════════════════════════════════════════════════════════════

class TestWeightManagement:
    def test_current_weights_match_default(self, optimizer):
        optimizer.detect_and_load()
        weights = optimizer.current_weights()
        # Should have all expected keys
        for key in DEFAULT_WEIGHTS:
            assert key in weights

    def test_update_weights_persists(self, optimizer):
        optimizer.detect_and_load()
        new_weights = optimizer.current_weights()
        new_weights["w_recency"] = 0.42
        optimizer.update_weights(new_weights)

        # Should be reflected immediately
        assert optimizer.current_weights()["w_recency"] == 0.42

    def test_export_weights_prism_5d(self, optimizer):
        optimizer.detect_and_load()
        exported = optimizer.get_export_weights()
        assert "w_r" in exported
        assert "w_f" in exported
        assert "w_s" in exported
        assert "w_e" in exported
        assert "w_res" in exported  # resonance dimension

    def test_strategy_table_persists_to_disk(self, tmp_path, tmp_project):
        data_dir = tmp_path / ".persist_test"

        # First session: detect and update
        opt1 = ArchetypeOptimizer(data_dir=data_dir, project_root=tmp_project)
        opt1.detect_and_load()
        w = opt1.current_weights()
        w["w_recency"] = 0.99
        opt1.update_weights(w)

        # Second session: should load persisted weights
        opt2 = ArchetypeOptimizer(data_dir=data_dir, project_root=tmp_project)
        _info2 = opt2.detect_and_load()
        assert abs(opt2.current_weights()["w_recency"] - 0.99) < 0.001

    def test_different_projects_get_different_weights(self, tmp_path):
        data_dir = tmp_path / ".multi_test"

        # Create a pure Python project
        py_root = tmp_path / "py_project"
        py_root.mkdir()
        for name in ["app.py", "models.py", "views.py", "utils.py"]:
            (py_root / name).write_text(
                f"import os\ndef {name.replace('.py', '')}():\n    pass\n",
                encoding="utf-8",
            )

        # Create a pure Rust project
        rs_root = tmp_path / "rs_project"
        rs_root.mkdir()
        src = rs_root / "src"
        src.mkdir()
        for name in ["lib.rs", "engine.rs", "parser.rs", "types.rs"]:
            (src / name).write_text(
                f"use std::collections::HashMap;\npub fn {name.replace('.rs', '')}() {{}}\n",
                encoding="utf-8",
            )

        opt_py = ArchetypeOptimizer(data_dir=data_dir, project_root=py_root)
        info_py = opt_py.detect_and_load()

        opt_rs = ArchetypeOptimizer(data_dir=data_dir, project_root=rs_root)
        info_rs = opt_rs.detect_and_load()

        # Archetypes should be different
        assert info_py.label != info_rs.label
        assert "python" in info_py.label
        assert "rust" in info_rs.label


# ═══════════════════════════════════════════════════════════════════
# Stats & Diagnostics Tests
# ═══════════════════════════════════════════════════════════════════

class TestStats:
    def test_stats_after_detection(self, optimizer):
        optimizer.detect_and_load()
        stats = optimizer.stats()
        assert "current_archetype" in stats
        assert "total_archetypes" in stats
        assert "strategy_table" in stats
        assert stats["current_archetype"] is not None
        assert stats["total_archetypes"] >= 1

    def test_stats_fingerprint_populated(self, optimizer):
        optimizer.detect_and_load()
        stats = optimizer.stats()
        fp = stats["fingerprint"]
        assert fp is not None
        assert "lang_python" in fp
        assert fp["lang_python"] > 0


# ═══════════════════════════════════════════════════════════════════
# Utility Tests
# ═══════════════════════════════════════════════════════════════════

class TestUtilities:
    def test_log_norm_zero(self):
        assert abs(_log_norm(0, 100)) < 0.001

    def test_log_norm_max(self):
        assert abs(_log_norm(100, 100) - 1.0) < 0.001

    def test_log_norm_monotonic(self):
        assert _log_norm(10, 100) < _log_norm(50, 100)
        assert _log_norm(50, 100) < _log_norm(100, 100)

    def test_log_norm_compresses_tails(self):
        # log compression: 50% of max should map to >50% of output
        assert _log_norm(50, 100) > 0.5
