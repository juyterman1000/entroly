from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CHILD = r"""
import json
from pathlib import Path
import sys
package_root, source_root, mode = map(str, sys.argv[1:])
sys.path.insert(0, package_root)
import entroly.repository_intelligence.service as module
assert Path(module.__file__).resolve() == Path(package_root) / 'entroly/repository_intelligence/service.py'
assert not sys.flags.ignore_environment
service = module.RepositoryIntelligenceService(source_root)
summary = service.summary()
index = service._index
context = service.context('caller', token_budget=4096)
plan = service.rename_preview('scale', 'rescale') if mode == 'resolved' else None
print(json.dumps({
    'index': index.to_dict(), 'index_digest': summary['index_digest'],
    'context_commitment': context['receipt']['context_sha256'],
    'plan_commitment': plan['receipt']['plan_sha256'] if plan else None,
    'hash_probe': hash('call-evidence-hash-seed-probe'),
}))
"""


@pytest.mark.parametrize("mode", ("resolved", "ambiguous"))
def test_same_line_call_sites_have_stable_evidence_and_commitments_across_processes(tmp_path: Path, mode):
    if mode == "resolved":
        (tmp_path / "source.py").write_text(
            "def scale(value):\n    return value\n"
            "def caller():\n    return scale(1) + scale(2) + scale(3)\n",
            encoding="utf-8",
        )
    else:
        (tmp_path / "first.py").write_text("def shared(value):\n    return value\n", encoding="utf-8")
        (tmp_path / "second.py").write_text("def shared(value):\n    return value\n", encoding="utf-8")
        (tmp_path / "source.py").write_text(
            "def caller():\n    return shared(1) + shared(2) + shared(3)\n",
            encoding="utf-8",
        )
    package_root = Path(__file__).resolve().parents[1]
    rows = []
    for seed in (0, 1, 7, 31):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(PYTHONHASHSEED=str(seed), PYTHONDONTWRITEBYTECODE="1",
                   ENTROLY_NO_SELF_HEAL="1", ENTROLY_AIR_GAP="1")
        child = subprocess.run(
            [sys.executable, "-B", "-s", "-c", CHILD, str(package_root), str(tmp_path), mode],
            cwd=tmp_path, env=env, text=True, capture_output=True, check=True, timeout=20,
        )
        assert not child.stderr
        rows.append(json.loads(child.stdout))
    assert len({row["hash_probe"] for row in rows}) > 1, "child seeds were not exercised"
    field = "call_edges" if mode == "resolved" else "unresolved_calls"
    expected_facts = None
    for row in rows:
        calls = row["index"][field]
        assert len(calls) == 3
        raw = (tmp_path / "source.py").read_bytes()
        spans = [(call["start_byte"], call["end_byte"]) for call in calls]
        assert len(set(spans)) == 3
        assert all(raw[start:end] for start, end in spans)
        assert all(hashlib.sha256(raw[call["start_byte"]:call["end_byte"]]).hexdigest()
                   == call["evidence_sha256"] for call in calls)
        facts = {json.dumps(call, sort_keys=True) for call in calls}
        if expected_facts is None:
            expected_facts = facts
        assert facts == expected_facts, "facts changed, rather than only their order"
        if mode == "ambiguous":
            assert row["index"]["call_edges"] == []
            assert all(call["reason"] == "ambiguous" and len(call["candidates"]) == 2 for call in calls)
    assert len({json.dumps(row["index"], sort_keys=True) for row in rows}) == 1
    assert len({row["index_digest"] for row in rows}) == 1
    assert len({row["context_commitment"] for row in rows}) == 1
    assert len({row["plan_commitment"] for row in rows}) == 1
