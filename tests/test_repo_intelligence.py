from __future__ import annotations

from entroly.repo_intelligence import RepositoryIntelligence


def test_python_impact_and_context_bundle_are_deterministic(tmp_path) -> None:
    package = tmp_path / "pkg"
    tests = tmp_path / "tests"
    package.mkdir()
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )
    (package / "service.py").write_text(
        "from .core import calculate_total\n\ndef checkout(items):\n    return calculate_total(items)\n",
        encoding="utf-8",
    )
    (tests / "test_service.py").write_text(
        "from pkg.service import checkout\n\ndef test_checkout():\n    assert checkout([1, 2]) == 3\n",
        encoding="utf-8",
    )

    intelligence = RepositoryIntelligence.scan(tmp_path)
    impact = intelligence.impact_report(["pkg/core.py"])
    impacted = [node.path for node in impact.impacted]
    assert impacted[0] == "pkg/core.py"
    assert "pkg/service.py" in impacted
    assert "tests/test_service.py" in impacted

    first = intelligence.context_bundle(
        query="checkout calculate total",
        changed_paths=["pkg/core.py"],
        budget_tokens=200,
    )
    second = intelligence.context_bundle(
        query="checkout calculate total",
        changed_paths=["pkg/core.py"],
        budget_tokens=200,
    )
    assert first.to_dict() == second.to_dict()
    assert "calculate_total" in first.render()
    assert first.emitted_tokens <= 200
    assert all(excerpt.start_line <= excerpt.end_line for excerpt in first.excerpts)


def test_workspace_escape_is_unresolved(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    intelligence = RepositoryIntelligence.scan(tmp_path)
    report = intelligence.impact_report(["../outside.py"])
    assert report.unresolved == ("../outside.py",)


def test_overview_importance_and_instruction_files(tmp_path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "core.py").write_text("def core():\n    return 1\n", encoding="utf-8")
    (pkg / "service.py").write_text(
        "from .core import core\n\ndef service():\n    return core()\n",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text(
        "Follow every security and release rule exactly.\n" * 20,
        encoding="utf-8",
    )
    intelligence = RepositoryIntelligence.scan(tmp_path)
    overview = intelligence.overview(top_n=5)
    ranks = dict(overview.top_important)
    assert ranks["pkg/core.py"] > ranks["pkg/service.py"]
    assert overview.instruction_files == ("AGENTS.md",)

    bundle = intelligence.context_bundle(
        query="security release rules",
        changed_paths=["AGENTS.md"],
        budget_tokens=500,
    )
    assert bundle.excerpts[0].start_line == 1
    assert bundle.excerpts[0].end_line == 20
    assert "instruction_file_full_fidelity" in bundle.excerpts[0].reasons


def test_smell_report_is_bounded_and_deterministic(tmp_path) -> None:
    body = "\n".join(f"    value_{index} = {index}" for index in range(90))
    (tmp_path / "large.py").write_text(
        "def oversized():\n" + body + "\n    return value_89\n",
        encoding="utf-8",
    )
    intelligence = RepositoryIntelligence.scan(tmp_path)
    first = intelligence.smell_report(max_findings=10)
    second = intelligence.smell_report(max_findings=10)
    assert first == second
    assert any(finding.kind == "long_function" for finding in first.findings)


def test_cjk_query_selects_relevant_symbol_excerpt(tmp_path) -> None:
    (tmp_path / "auth.py").write_text(
        'def authenticate():\n    """認証失敗を処理する。"""\n    return "認証失敗"\n',
        encoding="utf-8",
    )
    intelligence = RepositoryIntelligence.scan(tmp_path)
    bundle = intelligence.context_bundle(
        query="認証失敗の原因",
        changed_paths=["auth.py"],
        budget_tokens=100,
    )
    assert "認証失敗" in bundle.render()
