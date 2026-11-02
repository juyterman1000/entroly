"""Independent byte and runtime oracles for bounded Python re-export renames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from entroly.repository_intelligence import verified_refactor
from entroly.repository_intelligence.service import RepositoryIntelligenceService


CORE = "def compute(value):\n    return value + 2\n"
RENAMED_CORE = "def calculate(value):\n    return value + 2\n"


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*.py")
    }


def _plan(root: Path, origin: str = "core.py", *, max_changes: int = 10_000):
    service = RepositoryIntelligenceService(root)
    summary = service.summary()
    index = service._index
    assert index is not None
    targets = [
        symbol for symbol in index.symbols.values()
        if symbol.path == origin and symbol.name == "compute"
    ]
    assert len(targets) == 1
    before = _snapshot(root)
    plan = verified_refactor.build_verified_rename_plan(
        root, index, targets[0].symbol_id, "calculate",
        index_digest=summary["index_digest"], max_changes=max_changes,
    )
    assert plan["resolution"] == "resolved"
    assert plan["receipt"]["writes_performed"] == 0
    assert plan["receipt"]["remote_calls"] == 0
    assert plan["risk"]["reference_completeness"] == "not-proven"
    assert plan["risk"]["requires_incomplete_acknowledgement"] is True
    assert verified_refactor.verify_refactor_plan_commitment(plan)
    assert _snapshot(root) == before
    return index, summary["index_digest"], plan


def _closure(plan, status: str) -> None:
    report = plan["risk"]["python_export_closure"]
    assert report["status"] == status
    reasons = report["unknown_reasons"]
    assert isinstance(reasons, dict)
    assert all(
        isinstance(key, str) and key and type(count) is int and count > 0
        for key, count in reasons.items()
    )
    if status == "unknown":
        assert reasons, "uncertain exports need counted, auditable reasons"
    else:
        assert reasons == {}


def _assert_after_images(plan, before: dict[str, str], expected: dict[str, str]) -> None:
    """Replay only the public byte ranges; the oracle is authored separately."""
    staged = {name: source.encode("utf-8") for name, source in before.items()}
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for change in plan["changes"]:
        name = change["path"]
        assert name in before
        key = (name, change["start_byte"], change["end_byte"])
        assert key not in seen, "duplicate edit"
        seen.add(key)
        grouped.setdefault(name, []).append(change)
    assert plan["receipt"]["change_count"] == len(seen)
    for name, changes in grouped.items():
        raw = staged[name]
        cursor = len(raw)
        for change in sorted(changes, key=lambda item: item["start_byte"], reverse=True):
            start, end = change["start_byte"], change["end_byte"]
            assert 0 <= start < end <= cursor
            assert raw[start:end] == change["old_identifier"].encode("utf-8")
            assert change["old_identifier"] == "compute"
            assert change["new_identifier"] == "calculate"
            assert change["evidence_sha256"] == hashlib.sha256(raw[start:end]).hexdigest()
            assert change["source_sha256"] == hashlib.sha256(raw).hexdigest()
            staged[name] = (
                staged[name][:start] + b"calculate" + staged[name][end:]
            )
            cursor = start
    assert staged == {name: text.encode("utf-8") for name, text in expected.items()}


def _probe(root: Path, modules: list[str], assertions: str):
    # -I avoids ambient PYTHONPATH; -B avoids stale bytecode across rapid applies.
    # Check every requested import comes from this particular fixture directory.
    script = """
import importlib
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
for name in json.loads(sys.argv[2]):
    module = importlib.import_module(name)
    Path(module.__file__).resolve(strict=True).relative_to(root)
import consumer
""" + assertions + "\nprint(json.dumps(consumer.run()))\n"
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(root), json.dumps(modules)],
        cwd=root, capture_output=True, text=True, timeout=10, check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert not child.stderr
    return json.loads(child.stdout)


def _exercise(
    tmp_path: Path, before: dict[str, str], after: dict[str, str],
    modules: list[str], before_probe: str, after_probe: str, expected_result,
    origin: str = "core.py",
) -> None:
    project, oracle = tmp_path / "project", tmp_path / "authored-after"
    _write(project, before)
    _write(oracle, after)
    assert _probe(project, modules, before_probe) == expected_result
    assert _probe(oracle, modules, after_probe) == expected_result
    index, digest, plan = _plan(project, origin)
    _assert_after_images(plan, before, after)
    _closure(plan, "resolved-static-subset")
    # Use the real public apply with its hash, acknowledgement, and binding
    # review intact. No forged receipts or monkeypatched safety checks.
    result = verified_refactor.apply_verified_rename_plan(
        project, index, plan, index_digest=digest,
        expected_plan_sha256=plan["receipt"]["plan_sha256"],
        acknowledge_incomplete=True,
    )
    assert result["remote_calls"] == 0
    assert _snapshot(project) == {
        name: text.encode("utf-8") for name, text in after.items()
    }
    assert _probe(project, modules, after_probe) == expected_result


def test_static_chain_updates_exports_callbacks_and_global_uses_only(tmp_path: Path):
    before = {
        "core.py": CORE,
        "facade.py": (
            "# café: compute is the public operation\n"
            "from core import compute\n__all__ = ['compute']\n"
            "callback = compute\nlabel = 'compute'\n"
        ),
        "facade2.py": "from facade import compute\n__all__ = ('compute',)\n",
        "consumer.py": (
            "from facade2 import compute\nimport outsider\ncallback = compute\n"
            "def parameter(compute):\n    return compute(3)\n"
            "def local(value):\n    compute = lambda n: n\n    return compute(value)\n"
            "def run():\n    return [compute(3), callback(3), "
            "parameter(lambda n: n * 7), local(3), outsider.run()]\n"
        ),
        "other.py": "def compute(value):\n    return value + 100\n",
        "outsider.py": (
            "from other import compute\ncallback = compute\n"
            "__all__ = ['compute']\ndef run():\n    return compute(3)\n"
        ),
    }
    after = {
        **before,
        "core.py": RENAMED_CORE,
        "facade.py": (
            "# café: compute is the public operation\n"
            "from core import calculate\n__all__ = ['calculate']\n"
            "callback = calculate\nlabel = 'compute'\n"
        ),
        "facade2.py": "from facade import calculate\n__all__ = ('calculate',)\n",
        "consumer.py": (
            "from facade2 import calculate\nimport outsider\ncallback = calculate\n"
            "def parameter(compute):\n    return compute(3)\n"
            "def local(value):\n    compute = lambda n: n\n    return compute(value)\n"
            "def run():\n    return [calculate(3), callback(3), "
            "parameter(lambda n: n * 7), local(3), outsider.run()]\n"
        ),
    }
    _exercise(
        tmp_path, before, after, ["core", "facade", "facade2", "consumer", "outsider"],
        "from facade2 import *\nassert compute(3) == 5",
        "from facade2 import *\nassert calculate(3) == 5\n"
        "import facade, facade2, core\n"
        "assert facade.callback is core.calculate\n"
        "assert facade.__all__ == ['calculate']\n"
        "assert facade2.__all__ == ('calculate',)\n"
        "assert not hasattr(facade2, 'compute')",
        [5, 5, 21, 3, 103],
    )


@pytest.mark.parametrize("boundary", ["first-hop", "second-hop", "same-spelling"])
def test_explicit_alias_stops_public_name_propagation(tmp_path: Path, boundary: str):
    if boundary == "first-hop":
        first = "from core import compute as public_compute\n__all__ = ['public_compute']\n"
        first_after = "from core import calculate as public_compute\n__all__ = ['public_compute']\n"
        second = "from facade import public_compute\n__all__ = ['public_compute']\n"
        second_after = second
        public = "public_compute"
    elif boundary == "second-hop":
        first = "from core import compute\n__all__ = ['compute']\n"
        first_after = "from core import calculate\n__all__ = ['calculate']\n"
        second = "from facade import compute as public_compute\n__all__ = ['public_compute']\n"
        second_after = "from facade import calculate as public_compute\n__all__ = ['public_compute']\n"
        public = "public_compute"
    else:
        first = "from core import compute as compute\n__all__ = ['compute']\n"
        first_after = "from core import calculate as compute\n__all__ = ['compute']\n"
        second = "from facade import compute\n__all__ = ['compute']\n"
        second_after = second
        public = "compute"
    consumer = (
        f"from facade2 import {public}\ncallback = {public}\n"
        f"def run():\n    return [{public}(3), callback(4)]\n"
    )
    before = {
        "core.py": CORE, "facade.py": first,
        "facade2.py": second, "consumer.py": consumer,
    }
    after = {
        **before, "core.py": RENAMED_CORE,
        "facade.py": first_after, "facade2.py": second_after,
    }
    export_probe = (
        f"from facade2 import *\nassert {public}(3) == 5\n"
        f"import facade2\nassert facade2.__all__ == ['{public}']"
    )
    _exercise(
        tmp_path, before, after, ["core", "facade", "facade2", "consumer"],
        export_probe, export_probe + f"\nimport core\nassert {public} is core.calculate",
        [5, 6],
    )


def test_relative_chain_through_package_init(tmp_path: Path):
    before = {
        "pkg/core.py": CORE,
        "pkg/facade.py": "from .core import compute\n__all__ = ['compute']\n",
        "pkg/__init__.py": "from .facade import compute\n__all__ = ['compute']\n",
        "consumer.py": (
            "from pkg import compute\ncallback = compute\n"
            "def run():\n    return [compute(3), callback(4)]\n"
        ),
    }
    after = {
        "pkg/core.py": RENAMED_CORE,
        "pkg/facade.py": "from .core import calculate\n__all__ = ['calculate']\n",
        "pkg/__init__.py": "from .facade import calculate\n__all__ = ['calculate']\n",
        "consumer.py": (
            "from pkg import calculate\ncallback = calculate\n"
            "def run():\n    return [calculate(3), callback(4)]\n"
        ),
    }
    _exercise(
        tmp_path, before, after, ["pkg.core", "pkg.facade", "pkg", "consumer"],
        "from pkg import *\nassert compute(3) == 5",
        "from pkg import *\nassert calculate(3) == 5\n"
        "import pkg\nassert pkg.__all__ == ['calculate']",
        [5, 6], origin="pkg/core.py",
    )


@pytest.mark.parametrize("facade", [
    pytest.param(
        "import os\nif os.environ.get('EXPORT_CORE'):\n    from core import compute\n"
        "else:\n    from other import compute\n__all__ = ['compute']\n",
        id="conditional-import",
    ),
    pytest.param(
        "from core import compute\nfrom other import compute\n__all__ = ['compute']\n",
        id="ambiguous-rebinding",
    ),
    pytest.param("from core import *\n__all__ = ['compute']\n", id="star-export"),
    pytest.param(
        "from core import compute\n__all__ = [name for name in globals() "
        "if not name.startswith('_')]\n",
        id="dynamic-all",
    ),
])
def test_uncertain_exports_report_unknown_without_guessing_downstream(tmp_path: Path, facade: str):
    files = {
        "core.py": CORE,
        "other.py": "def compute(value):\n    return value + 100\n",
        "facade.py": facade,
        "facade2.py": "from facade import compute\n__all__ = ['compute']\n",
        "consumer.py": (
            "from facade2 import compute\ncallback = compute\n"
            "def run():\n    return compute(3)\n"
        ),
    }
    _write(tmp_path, files)
    _index, _digest, plan = _plan(tmp_path)
    _closure(plan, "unknown")
    assert not any(
        change["path"] in {"facade2.py", "consumer.py", "other.py"}
        for change in plan["changes"]
    ), "an uncertain export must not authorize downstream name rewrites"
    # Direct verified edits may remain: unknown closure is not a universal veto.
    assert any(change["path"] == "core.py" for change in plan["changes"])
    assert _snapshot(tmp_path) == {
        name: text.encode("utf-8") for name, text in files.items()
    }


def test_global_import_alias_rebinding_is_not_an_export_origin(tmp_path: Path):
    files = {
        "core.py": CORE,
        "other.py": "VALUE = 103\n",
        "facade.py": (
            "from core import compute\n"
            "def retarget():\n    global compute\n    import other as compute\n"
            "retarget()\ncallback = compute\n__all__ = ['compute']\n"
        ),
        "facade2.py": "from facade import compute\n__all__ = ['compute']\n",
        "consumer.py": (
            "from facade2 import compute\ncallback = compute\n"
            "def run():\n    return callback.VALUE\n"
        ),
    }
    _write(tmp_path, files)
    assert _probe(tmp_path, ["facade", "facade2", "consumer"], "") == 103
    _index, _digest, plan = _plan(tmp_path)
    # The direct import's origin token may change; neither the module object
    # bound by the global import nor its downstream references may change.
    assert not any(
        change["path"] in {"facade2.py", "consumer.py", "other.py"}
        for change in plan["changes"]
    ), "a global import alias rebinds the apparent re-export to another object"
    start = files["facade.py"].encode().index(b"callback = compute") + len(b"callback = ")
    assert not any(
        change["path"] == "facade.py" and change["start_byte"] == start
        for change in plan["changes"]
    )
    _closure(plan, "unknown")


def test_module_exception_target_does_not_rename_its_exception_load(tmp_path: Path):
    before = {
        "core.py": CORE,
        "facade.py": (
            "from core import compute\ntry:\n    1 / 0\n"
            "except ZeroDivisionError as compute:\n    callback = compute\n"
        ),
        "consumer.py": "import facade\ndef run():\n    return str(facade.callback)\n",
    }
    after = {
        **before, "core.py": RENAMED_CORE,
        "facade.py": (
            "from core import calculate\ntry:\n    1 / 0\n"
            "except ZeroDivisionError as compute:\n    callback = compute\n"
        ),
    }
    project, oracle = tmp_path / "project", tmp_path / "authored-after"
    _write(project, before)
    _write(oracle, after)
    assert _probe(project, ["facade", "consumer"], "") == "division by zero"
    assert _probe(oracle, ["facade", "consumer"], "") == "division by zero"
    _index, _digest, plan = _plan(project)
    _assert_after_images(plan, before, after)
    _closure(plan, "unknown")


def test_conditional_export_contract_is_an_unknown_frontier(tmp_path: Path):
    files = {
        "core.py": CORE,
        "facade.py": (
            "from core import compute\nimport os\n"
            "if os.name:\n    __all__ = ['compute']\n"
        ),
        "facade2.py": "from facade import compute\n__all__ = ['compute']\n",
        "consumer.py": "from facade2 import compute\ndef run():\n    return compute(3)\n",
    }
    _write(tmp_path, files)
    _index, _digest, plan = _plan(tmp_path)
    _closure(plan, "unknown")
    assert not any(
        change["path"] in {"facade2.py", "consumer.py"}
        for change in plan["changes"]
    ), "a conditional export contract must not be promoted to a static closure"


def test_change_budget_cannot_leave_a_resolved_closure_with_dropped_edits(tmp_path: Path):
    _write(tmp_path, {
        "core.py": CORE,
        "facade.py": "from core import compute\n__all__ = ['compute']\n",
    })
    _index, _digest, plan = _plan(tmp_path, max_changes=2)
    assert 0 < len(plan["changes"]) <= 2
    assert plan["receipt"]["omissions_by_reason"].get("change-limit", 0) > 0
    _closure(plan, "unknown")
