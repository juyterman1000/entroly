"""Trust-boundary tests for the Exp 1 reproducibility artifact and result."""
from __future__ import annotations

import hashlib
import json

import pytest

from docs.research.exp1.capture_selection import (
    CORPUS_SCHEMA,
    _validate_artifact,
    canonical,
)
from docs.research.exp1.repro_harness import (
    _validate_capture,
    jaccard,
    kendall_tau,
    selection_keys,
)


def _artifact() -> dict:
    fragments = [
        {
            "source": "file:a.py",
            "content": "x = 1\n",
            "fragment_id": "f1",
            "feedback_multiplier": 1.0,
        }
    ]
    canonical_fragments = json.dumps(
        fragments,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema_version": CORPUS_SCHEMA,
        "metadata": {
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "entroly_version": "1.0.66",
            "native_version": "1.0.66",
            "native_module_sha256": "c" * 64,
            "fragments_sha256": hashlib.sha256(canonical_fragments).hexdigest(),
            "fragment_count": 1,
        },
        "fragments": fragments,
    }


def _selection(source_ids: list[str] | None = None) -> dict:
    return canonical(
        [
            {
                "source": "file:a.py",
                "content": "x = 1\n",
                "source_fragment_ids": source_ids or ["f1"],
            }
        ]
    )


def test_artifact_digest_and_count_are_verified():
    artifact = _artifact()
    fragments, metadata = _validate_artifact(artifact)
    assert len(fragments) == metadata["fragment_count"] == 1

    artifact["metadata"]["fragment_count"] = 2
    with pytest.raises(ValueError, match="fragment_count"):
        _validate_artifact(artifact)

    artifact["metadata"]["fragment_count"] = True
    with pytest.raises(ValueError, match="fragment_count"):
        _validate_artifact(artifact)


def test_artifact_source_and_runtime_identity_are_required():
    artifact = _artifact()
    artifact["metadata"]["source_commit"] = "abc123"
    with pytest.raises(ValueError, match="source_commit"):
        _validate_artifact(artifact)

    artifact = _artifact()
    artifact["metadata"]["native_module_sha256"] = ""
    with pytest.raises(ValueError, match="identity"):
        _validate_artifact(artifact)


def test_selection_digest_binds_origin_ids_and_utf8_byte_length():
    first = _selection(["f1"])
    second = _selection(["f2"])
    assert first["digest"] != second["digest"]

    unicode_result = canonical(
        [{"source": "file:u.py", "content": "漢", "source_fragment_ids": ["u1"]}]
    )
    assert unicode_result["order"][0]["content_len"] == len("漢".encode())


def test_capture_contract_recomputes_digest_and_rejects_tampering():
    result = _selection()
    assert _validate_capture(result) == result
    result["order"][0]["source"] = "file:tampered.py"
    with pytest.raises(RuntimeError, match="digest"):
        _validate_capture(result)


def test_fragment_identity_metrics_do_not_collapse_same_source_changes():
    left = _selection(["f1"])
    right = _selection(["f2"])
    left_keys = selection_keys(left)
    right_keys = selection_keys(right)
    assert jaccard(left_keys, right_keys) == 0.0
    assert kendall_tau(left_keys, right_keys) is None


def test_duplicate_fragment_occurrences_remain_distinct():
    item = _selection()["order"][0]
    result = {
        "order": [
            {**item, "rank": 0},
            {**item, "rank": 1},
        ]
    }
    keys = selection_keys(result)
    assert len(keys) == len(set(keys)) == 2


def test_repro_harness_writes_machine_readable_valid_result(
    tmp_path,
    monkeypatch,
):
    from docs.research.exp1 import repro_harness

    fragments = [
        {
            "source": f"file:{name}.py",
            "content": name,
            "fragment_id": f"id-{name}",
        }
        for name in ("b", "a")
    ]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "schema_version": repro_harness.CORPUS_SCHEMA,
                "metadata": {"test": True},
                "fragments": fragments,
            }
        ),
        encoding="utf-8",
    )

    def fake_capture(path, _env):
        with open(path, encoding="utf-8") as source:
            artifact = json.load(source)
        selected = [
            {
                **fragment,
                "source_fragment_ids": [fragment["fragment_id"]],
            }
            for fragment in sorted(
                artifact["fragments"],
                key=lambda item: item["source"],
            )
        ]
        return canonical(selected)

    monkeypatch.setattr(repro_harness, "CORPUS", str(corpus))
    monkeypatch.setattr(repro_harness, "run_capture", fake_capture)
    output = tmp_path / "result.json"
    assert repro_harness.main(str(output)) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert result["verdict"] == {
        "complete": True,
        "set_identical": True,
        "byte_identical": True,
    }
