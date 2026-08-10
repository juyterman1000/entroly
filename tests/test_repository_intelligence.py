from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly.repository_intelligence import (
    RepositoryLimits,
    analyze_change_impact,
    build_repository_index,
    localize_tests,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> None:
    _write(root, "app/__init__.py", "")
    _write(
        root,
        "app/service.py",
        "def charge_card(customer, amount):\n"
        "    return gateway_charge(customer.card, amount)\n\n"
        "def gateway_charge(card, amount):\n"
        "    return {'status': 'ok', 'amount': amount}\n",
    )
    _write(
        root,
        "app/api.py",
        "from app.service import charge_card\n\n"
        "def checkout(customer, amount):\n"
        "    return charge_card(customer, amount)\n",
    )
    _write(
        root,
        "tests/test_api.py",
        "from app.api import checkout\n\n"
        "def test_checkout():\n"
        "    assert checkout(object(), 10)\n",
    )
    _write(
        root,
        "tests/test_unrelated.py",
        "def test_unrelated():\n    assert 1 + 1 == 2\n",
    )


def test_python_symbols_imports_calls_impact_and_tests(tmp_path: Path) -> None:
    _project(tmp_path)
    index = build_repository_index(tmp_path)

    service = {symbol.name: symbol for symbol in index.symbols_for_path("app/service.py")}
    api = {symbol.name: symbol for symbol in index.symbols_for_path("app/api.py")}
    assert {"charge_card", "gateway_charge"}.issubset(service)
    assert service["charge_card"].line_start == 1
    assert service["charge_card"].line_end == 2
    assert index.file_dependencies["app/api.py"] == ("app/service.py",)
    assert index.file_dependencies["tests/test_api.py"] == ("app/api.py",)

    edges = {(edge.caller_id, edge.callee_id) for edge in index.call_edges}
    assert (api["checkout"].symbol_id, service["charge_card"].symbol_id) in edges

    impact = analyze_change_impact(index, ["app/service.py"])
    assert impact.impacted_paths == (
        "app/api.py",
        "app/service.py",
        "tests/test_api.py",
    )
    assert any(
        reason.startswith("imports:app/service.py")
        for reason in impact.reasons["app/api.py"]
    )

    candidates = localize_tests(index, ["app/service.py"])
    assert candidates[0].path == "tests/test_api.py"
    assert all(candidate.path != "tests/test_unrelated.py" for candidate in candidates)
    assert candidates[0].score >= 50


def test_class_methods_and_exact_ranges(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/model.py",
        "class Account:\n"
        "    def debit(self, amount):\n"
        "        if amount <= 0:\n"
        "            raise ValueError(amount)\n"
        "        return amount\n",
    )
    index = build_repository_index(tmp_path)
    symbols = {symbol.qualified_name: symbol for symbol in index.symbols.values()}
    assert symbols["Account"].kind == "class"
    assert symbols["Account.debit"].kind == "method"
    assert symbols["Account.debit"].line_start == 2
    assert symbols["Account.debit"].line_end == 5
    assert symbols["Account.debit"].parent_id == symbols["Account"].symbol_id


def test_parse_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    _write(tmp_path, "broken.py", "def nope(:\n")
    _write(tmp_path, "good.py", "def ok():\n    return True\n")
    index = build_repository_index(tmp_path)
    assert index.files["broken.py"].parse_error.startswith("SyntaxError:")
    assert [symbol.name for symbol in index.symbols_for_path("good.py")] == ["ok"]


def test_rust_and_typescript_conservative_symbols(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/lib.rs",
        "pub fn compress(input: &str) -> usize { helper(input) }\n"
        "fn helper(_: &str) -> usize { 1 }\n",
    )
    _write(
        tmp_path,
        "web/index.ts",
        "export function render(value: string) { return compress(value); }\n",
    )
    index = build_repository_index(tmp_path)
    assert {
        symbol.name for symbol in index.symbols_for_path("src/lib.rs")
    } == {"compress", "helper"}
    assert {symbol.name for symbol in index.symbols_for_path("web/index.ts")} == {"render"}
    assert any(edge.callee_id.endswith("::compress::fn") for edge in index.call_edges)


def test_deterministic_json_contract(tmp_path: Path) -> None:
    _project(tmp_path)
    first = build_repository_index(tmp_path).to_dict()
    # Rewrite in a different filesystem order without changing bytes.
    for path in sorted(tmp_path.rglob("*.py"), reverse=True):
        data = path.read_bytes()
        path.unlink()
        path.write_bytes(data)
    second = build_repository_index(tmp_path).to_dict()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_limits_are_explicit(tmp_path: Path) -> None:
    for index in range(4):
        _write(tmp_path, f"f{index}.py", f"def f{index}():\n    return {index}\n")
    result = build_repository_index(tmp_path, limits=RepositoryLimits(max_files=2))
    assert len(result.files) == 2
    assert "repository limits reached; index truncated" in result.diagnostics


def test_escaping_symlink_is_ignored(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("def secret():\n    return 1\n", encoding="utf-8")
    link = tmp_path / "escape.py"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    index = build_repository_index(tmp_path)
    assert "escape.py" not in index.files
    assert any("unsafe or unreadable" in item for item in index.diagnostics)


def test_impact_truncation_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    for i in range(5):
        _write(tmp_path, f"d{i}.py", f"from a import a\ndef d{i}():\n    return a()\n")
    index = build_repository_index(tmp_path)
    report = analyze_change_impact(index, ["a.py"], max_impacted_paths=2)
    assert report.truncated
    assert len(report.impacted_paths) == 2


def test_direct_import_alias_resolves_without_global_guessing(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/source.py", "def execute():\n    return 1\n")
    _write(
        tmp_path,
        "pkg/caller.py",
        "from pkg.source import execute as run\n"
        "def invoke():\n"
        "    return run()\n",
    )
    index = build_repository_index(tmp_path)
    source = index.symbols_for_path("pkg/source.py")[0]
    caller = index.symbols_for_path("pkg/caller.py")[0]
    assert any(
        edge.caller_id == caller.symbol_id and edge.callee_id == source.symbol_id
        for edge in index.call_edges
    )


def test_ambiguous_symbol_name_does_not_invent_call_edge(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def execute():\n    return 'a'\n")
    _write(tmp_path, "b.py", "def execute():\n    return 'b'\n")
    _write(tmp_path, "caller.py", "def caller():\n    return execute()\n")
    index = build_repository_index(tmp_path)
    caller_id = index.symbols_for_path("caller.py")[0].symbol_id
    assert all(edge.caller_id != caller_id for edge in index.call_edges)
    unresolved = [call for call in index.unresolved_calls if call.caller_id == caller_id]
    assert len(unresolved) == 1
    assert unresolved[0].reason == "ambiguous"
    assert len(unresolved[0].candidates) == 2


def test_python_receiver_annotation_resolves_exact_class_member(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "transport.py",
        "class HttpClient:\n"
        "    def send(self):\n        return 'http'\n\n"
        "class MailClient:\n"
        "    def send(self):\n        return 'mail'\n",
    )
    _write(
        tmp_path,
        "service.py",
        "from transport import HttpClient, MailClient\n\n"
        "def deliver(client: MailClient):\n"
        "    return client.send()\n",
    )
    index = build_repository_index(tmp_path)
    symbols = {symbol.qualified_name: symbol for symbol in index.symbols.values()}
    edge = next(edge for edge in index.call_edges if edge.caller_id == symbols["deliver"].symbol_id)
    assert edge.callee_id == symbols["MailClient.send"].symbol_id
    assert edge.confidence == "type-inferred"
    assert edge.resolution == "receiver-type:annotation"


def test_python_constructor_assignment_resolves_cross_file_member(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "client.py",
        "class Client:\n"
        "    def fetch(self):\n        return 1\n",
    )
    _write(
        tmp_path,
        "use.py",
        "from client import Client\n\n"
        "def run():\n"
        "    client = Client()\n"
        "    return client.fetch()\n",
    )
    index = build_repository_index(tmp_path)
    symbols = {symbol.qualified_name: symbol for symbol in index.symbols.values()}
    edge = next(
        edge
        for edge in index.call_edges
        if edge.caller_id == symbols["run"].symbol_id
        and edge.callee_id == symbols["Client.fetch"].symbol_id
    )
    assert edge.resolution == "receiver-type:constructor-assignment"


def test_python_self_call_resolves_only_enclosing_class_member(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "workers.py",
        "class First:\n"
        "    def execute(self):\n        return 1\n"
        "    def run(self):\n        return self.execute()\n\n"
        "class Second:\n"
        "    def execute(self):\n        return 2\n",
    )
    index = build_repository_index(tmp_path)
    symbols = {symbol.qualified_name: symbol for symbol in index.symbols.values()}
    edge = next(edge for edge in index.call_edges if edge.caller_id == symbols["First.run"].symbol_id)
    assert edge.callee_id == symbols["First.execute"].symbol_id
    assert edge.resolution == "receiver-type:implicit-self"


def test_untyped_member_call_does_not_bind_same_file_method_by_name(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "unsafe.py",
        "class Unrelated:\n"
        "    def get(self):\n        return 1\n\n"
        "def run(client):\n"
        "    return client.get()\n",
    )
    index = build_repository_index(tmp_path)
    run_id = next(
        symbol.symbol_id for symbol in index.symbols.values() if symbol.qualified_name == "run"
    )
    assert all(edge.caller_id != run_id for edge in index.call_edges)
    unresolved = [
        call for call in index.unresolved_calls if call.caller_id == run_id
    ]
    assert len(unresolved) == 1
    assert unresolved[0].reason == "untyped-receiver-member"
    assert len(unresolved[0].candidates) == 1
    assert unresolved[0].candidates[0].endswith("::Unrelated.get::method")


def test_tree_sitter_call_is_attributed_to_narrowest_rust_symbol(tmp_path: Path) -> None:
    # Presence is not compatibility. An installed pack below the declared
    # floor still imports, so `importorskip` lets the test run against a
    # registry whose symbol extraction differs, and it fails with a confusing
    # KeyError rather than skipping. Gate on the same compatibility check the
    # runtime uses so a below-floor environment skips honestly.
    pytest.importorskip("tree_sitter_language_pack")
    from entroly.parser_compatibility import language_pack_status

    status = language_pack_status()
    if not status.compatible:
        pytest.skip(
            f"tree-sitter-language-pack {status.version} is below the supported "
            f">={status.minimum_version} floor ({status.detail})"
        )
    _write(
        tmp_path,
        "src/lib.rs",
        "fn helper() {}\n"
        "struct Worker;\n"
        "impl Worker {\n"
        "    fn run(&self) { helper(); }\n"
        "}\n",
    )
    index = build_repository_index(tmp_path)
    symbols = {symbol.qualified_name: symbol for symbol in index.symbols.values()}
    edge = next(edge for edge in index.call_edges if edge.callee_id == symbols["helper"].symbol_id)
    assert edge.caller_id == symbols["Worker.run"].symbol_id
    assert edge.resolution == "same-file"
    assert edge.start_byte < edge.end_byte
    assert len(edge.evidence_sha256) == 64


def test_changed_seed_limit_is_enforced(tmp_path: Path) -> None:
    for index in range(4):
        _write(tmp_path, f"f{index}.py", f"def f{index}():\n    return {index}\n")
    repository = build_repository_index(tmp_path)
    report = analyze_change_impact(
        repository,
        [f"f{index}.py" for index in range(4)],
        max_impacted_paths=2,
    )
    assert report.truncated
    assert report.changed_paths == ("f0.py", "f1.py")
    assert report.impacted_paths == ("f0.py", "f1.py")


def test_call_edge_limit_is_global_not_per_file(tmp_path: Path) -> None:
    _write(tmp_path, "target.py", "def target():\n    return 1\n")
    for index in range(5):
        _write(
            tmp_path,
            f"caller_{index}.py",
            "from target import target\n"
            f"def caller_{index}():\n"
            "    return target()\n",
        )
    repository = build_repository_index(
        tmp_path,
        limits=RepositoryLimits(max_edges=2),
    )
    assert len(repository.call_edges) == 2


def test_relationship_limit_also_bounds_ambiguous_call_evidence(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", "def duplicate():\n    return 'a'\n")
    _write(tmp_path, "b.py", "def duplicate():\n    return 'b'\n")
    for index in range(10):
        _write(
            tmp_path,
            f"caller_{index}.py",
            f"def caller_{index}():\n    return duplicate()\n",
        )
    repository = build_repository_index(
        tmp_path,
        limits=RepositoryLimits(max_edges=3),
    )
    assert len(repository.call_edges) + len(repository.unresolved_calls) == 3
    assert "relationship limit reached; remaining evidence omitted" in repository.diagnostics


def test_index_serialization_is_explicitly_versioned(tmp_path: Path) -> None:
    _write(tmp_path, "sample.py", "def sample():\n    return 1\n")
    payload = build_repository_index(tmp_path).to_dict()
    assert payload["schema_version"] == "entroly.repository-index.v2"
