"""A rename must expose compiler conflicts and reject them before writes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence import verified_refactor, verified_refactor_impl


CALLERS = {
    "parameter": "from payments import charge\ndef process(authorize):\n    return charge(authorize)\n",
    "existing-binding": "from payments import charge\nauthorize = lambda amount: amount * 7\ndef process(amount):\n    return charge(amount), authorize(amount)\n",
    "callback": "from payments import charge\nhandler = charge\ndef process(amount):\n    return handler(amount)\n",
    "direct": "from payments import charge\ndef process(amount):\n    return charge(amount)\n",
    "alias": "from payments import charge as invoke\ndef process(authorize):\n    return invoke(authorize)\n",
    "same-spelling-alias": "from payments import charge as charge\ndef process(amount):\n    return charge(amount)\n",
    "unrelated-local": "from payments import charge\ndef other(authorize):\n    return authorize\ndef process(amount):\n    return charge(amount)\n",
    "attribute": "import payments\ndef process(amount):\n    return payments.charge(amount)\n",
}


def _project(root: Path, kind: str):
    (root / "payments.py").write_text(
        "def charge(amount):\n    return amount + 2\n", encoding="utf-8"
    )
    (root / "consumer.py").write_text(CALLERS[kind], encoding="utf-8")
    service = RepositoryIntelligenceService(root)
    summary = service.summary()
    index = service._index
    assert index is not None
    plan = verified_refactor.build_verified_rename_plan(
        root, index, "charge", "authorize", index_digest=summary["index_digest"]
    )
    assert plan["resolution"] == "resolved"
    return index, summary["index_digest"], plan


def _bytes(root: Path):
    return {path.name: path.read_bytes() for path in root.glob("*.py")}


def _apply(root, index, digest, plan, implementation=verified_refactor):
    return implementation.apply_verified_rename_plan(
        root, index, plan, index_digest=digest,
        expected_plan_sha256=plan["receipt"]["plan_sha256"],
        acknowledge_incomplete=True,
    )


@pytest.mark.parametrize("kind", ("parameter", "existing-binding"))
def test_preview_exposes_source_bound_conflicts_without_writes(tmp_path: Path, kind: str):
    _index, _digest, plan = _project(tmp_path, kind)
    before = _bytes(tmp_path)
    review = plan["binding_review"]
    assert review["status"] == "conflict"
    assert review["conflict_count"] > 0
    assert review["semantic_status"] == "not-checked"
    assert plan["risk"]["reference_completeness"] == "not-proven"
    assert plan["receipt"]["writes_performed"] == 0
    assert verified_refactor.verify_refactor_plan_commitment(plan)
    for file_review in review["files"]:
        raw = (tmp_path / file_review["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == file_review["source_sha256"]
        for conflict in file_review["conflicts"]:
            token = raw[conflict["start_byte"]:conflict["end_byte"]]
            assert token == conflict["identifier"].encode()
            assert hashlib.sha256(token).hexdigest() == conflict["evidence_sha256"]
    assert _bytes(tmp_path) == before


@pytest.mark.parametrize("kind", ("parameter", "existing-binding"))
@pytest.mark.parametrize("implementation", (verified_refactor, verified_refactor_impl))
def test_apply_rejects_conflicts_before_any_write(tmp_path: Path, kind: str, implementation):
    index, digest, plan = _project(tmp_path, kind)
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match="binding conflict"):
        _apply(tmp_path, index, digest, plan, implementation)
    assert _bytes(tmp_path) == before


@pytest.mark.parametrize("remove_review", (True, False))
def test_apply_recomputes_instead_of_trusting_review_metadata(tmp_path: Path, remove_review: bool):
    index, digest, plan = _project(tmp_path, "parameter")
    forged = copy.deepcopy(plan)
    if remove_review:
        forged.pop("binding_review", None)
    else:
        forged["binding_review"] = {"status": "clear", "conflict_count": 0, "files": []}
    forged["receipt"].pop("plan_sha256")
    forged = verified_refactor_impl._finish_plan(forged)
    assert verified_refactor.verify_refactor_plan_commitment(forged)
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match="binding conflict"):
        _apply(tmp_path, index, digest, forged)
    assert _bytes(tmp_path) == before


def _run(root: Path):
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", """
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
import consumer
assert Path(consumer.__file__).resolve(strict=True) == root / 'consumer.py'
print(json.dumps(consumer.process(3)))
""", str(root)], cwd=root, capture_output=True, text=True, check=True, timeout=10,
    )
    assert not child.stderr
    return json.loads(child.stdout)


@pytest.mark.parametrize("kind", ("direct", "alias", "same-spelling-alias", "unrelated-local", "callback"))
def test_nonconflicting_rename_still_completes_with_same_caller_result(tmp_path: Path, kind: str):
    index, digest, plan = _project(tmp_path, kind)
    assert _run(tmp_path) == 5
    assert plan["binding_review"]["conflict_count"] == 0
    assert plan["binding_review"]["semantic_status"] == "not-checked"
    result = _apply(tmp_path, index, digest, plan)
    assert result["file_count"] == 2
    assert "def authorize" in (tmp_path / "payments.py").read_text(encoding="utf-8")
    assert _run(tmp_path) == 5


@pytest.mark.parametrize("implementation", (verified_refactor, verified_refactor_impl))
def test_partial_callback_plan_still_rejected_before_writes(tmp_path: Path, implementation):
    index, digest, plan = _project(tmp_path, "callback")
    plan["changes"] = [item for item in plan["changes"]
                       if item["kind"] != "python-module-reference"]
    plan["receipt"]["change_count"] = len(plan["changes"])
    plan["receipt"].pop("plan_sha256")
    plan = verified_refactor_impl._finish_plan(plan)
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match="binding conflict"):
        _apply(tmp_path, index, digest, plan, implementation)
    assert _bytes(tmp_path) == before


def test_unknown_attribute_scope_is_not_mislabeled_safe(tmp_path: Path):
    index, digest, plan = _project(tmp_path, "attribute")
    review = plan["binding_review"]
    assert review["status"] == "unknown"
    assert review["semantic_status"] == "not-checked"
    assert review["conflict_count"] == 0
    assert plan["risk"]["requires_incomplete_acknowledgement"] is True
    assert _run(tmp_path) == 5
    _apply(tmp_path, index, digest, plan)
    assert _run(tmp_path) == 5


@pytest.mark.parametrize("implementation", (verified_refactor, verified_refactor_impl))
def test_partial_recursive_rename_cannot_reuse_a_clear_preview(tmp_path: Path, implementation):
    source = tmp_path / "payments.py"
    source.write_text(
        "def charge(amount):\n    if amount <= 0:\n        return 2\n    return charge(amount - 1)\n",
        encoding="utf-8",
    )
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    index = service._index
    assert index is not None
    plan = verified_refactor.build_verified_rename_plan(
        tmp_path, index, "charge", "authorize", index_digest=summary["index_digest"],
    )
    assert len(plan["changes"]) == 2
    plan["changes"] = [item for item in plan["changes"] if item["kind"] == "definition"]
    plan["receipt"]["change_count"] = len(plan["changes"])
    plan["receipt"].pop("plan_sha256")
    plan = verified_refactor_impl._finish_plan(plan)
    assert verified_refactor.verify_refactor_plan_commitment(plan)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="binding conflict"):
        _apply(tmp_path, index, summary["index_digest"], plan, implementation)
    assert source.read_bytes() == before


def test_compiler_rejected_target_never_reaches_filesystem_writes(tmp_path: Path):
    index, digest, _plan = _project(tmp_path, "direct")
    plan = verified_refactor.build_verified_rename_plan(
        tmp_path, index, "charge", "__debug__", index_digest=digest,
    )
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match="compiler|syntax"):
        _apply(tmp_path, index, digest, plan)
    assert _bytes(tmp_path) == before


def test_json_round_trip_preserves_commitment_and_successful_apply(tmp_path: Path):
    index, digest, plan = _project(tmp_path, "direct")
    decoded = json.loads(json.dumps(plan))
    assert verified_refactor.verify_refactor_plan_commitment(decoded)
    _apply(tmp_path, index, digest, decoded)
    assert _run(tmp_path) == 5


def test_preview_file_limit_does_not_skip_apply_review(tmp_path: Path, monkeypatch):
    (tmp_path / "payments.py").write_text(
        "def charge(amount):\n return amount + 2\n", encoding="utf-8",
    )
    (tmp_path / "aaa.py").write_text(CALLERS["direct"], encoding="utf-8")
    (tmp_path / "zzz.py").write_text(CALLERS["parameter"], encoding="utf-8")
    monkeypatch.setattr(verified_refactor_impl, "MAX_PREVIEW_REVIEW_FILES", 1)
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    index = service._index
    assert index is not None
    plan = verified_refactor.build_verified_rename_plan(
        tmp_path, index, "charge", "authorize", index_digest=summary["index_digest"],
    )
    review = plan["binding_review"]
    assert review["reviewed_file_count"] == 1
    assert review["omitted_file_count"] == 2
    assert review["status"] == "unknown"
    assert review["conflict_count"] == 0
    before = _bytes(tmp_path)
    with pytest.raises(ValueError, match="binding conflict"):
        _apply(tmp_path, index, summary["index_digest"], plan)
    assert _bytes(tmp_path) == before
