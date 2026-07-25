from __future__ import annotations

import io
import json
import tarfile

import pytest

from benchmarks.contextbench_pilot import (
    _download,
    _extract_stripped,
    _records_to_spans,
    _selection_digest,
)
from benchmarks.contextbench_span_adapter import SelectedSpan


def _archive(*members: tuple[str, bytes, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload.decode()
                archive.addfile(info)
            else:
                raise AssertionError(kind)
    return buffer.getvalue()


def test_safe_archive_extracts_regular_files(tmp_path):
    data = _archive(("repo-sha/src/main.py", b"print('ok')\n", "file"))
    assert _extract_stripped(data, str(tmp_path)) == 1
    assert (tmp_path / "src" / "main.py").read_bytes() == b"print('ok')\n"


@pytest.mark.parametrize(
    "name",
    [
        "repo-sha/../../outside.txt",
        "/absolute/repo-sha/file.txt",
        "repo-sha\\..\\outside.txt",
    ],
)
def test_archive_path_escape_is_rejected(tmp_path, name):
    data = _archive((name, b"owned", "file"))
    with pytest.raises(ValueError, match="unsafe"):
        _extract_stripped(data, str(tmp_path))


def test_archive_links_are_rejected(tmp_path):
    data = _archive(("repo-sha/link", b"../../outside", "symlink"))
    with pytest.raises(ValueError, match="links"):
        _extract_stripped(data, str(tmp_path))


def test_archive_with_multiple_roots_is_rejected(tmp_path):
    data = _archive(
        ("repo-a/a.py", b"a", "file"),
        ("repo-b/b.py", b"b", "file"),
    )
    with pytest.raises(ValueError, match="top-level roots"):
        _extract_stripped(data, str(tmp_path))


def test_download_rejects_non_github_urls_and_short_revisions():
    with pytest.raises(ValueError, match="github.com"):
        _download("https://example.com/owner/repo", "a" * 40)
    with pytest.raises(ValueError, match="40-character"):
        _download("https://github.com/owner/repo", "abc123")
    with pytest.raises(ValueError, match="canonical"):
        _download("https://user@github.com/owner/repo", "a" * 40)


def test_same_file_records_are_merged_not_overwritten():
    records = [
        SelectedSpan(
            path="a.py",
            score=2.0,
            rank=0,
            token_cost=2,
            lines={1, 2},
            mapped=True,
            mapped_blocks=1,
        ),
        SelectedSpan(
            path="a.py",
            score=1.0,
            rank=1,
            token_cost=2,
            lines={5, 6},
            mapped=True,
            mapped_blocks=1,
        ),
    ]
    assert _records_to_spans(records) == {"a.py": [[1, 2], [5, 6]]}


def test_selection_digest_covers_partial_attribution_failures():
    clean = SelectedSpan(
        path="a.py",
        score=1.0,
        rank=0,
        token_cost=2,
        lines={1, 2},
        mapped=True,
        mapped_blocks=1,
    )
    partial = SelectedSpan(
        path="a.py",
        score=1.0,
        rank=0,
        token_cost=2,
        lines={1, 2},
        mapped=True,
        reason="not_found",
        mapped_blocks=1,
        unmapped_blocks=1,
        unmapped_lines=3,
    )
    assert _selection_digest([clean]) != _selection_digest([partial])


def _task() -> dict:
    return {
        "instance_id": "task-1",
        "repo_url": "https://github.com/owner/repo",
        "base_commit": "a" * 40,
        "problem_statement": "find the parser",
        "gold_context": json.dumps(
            [{"file": "a.py", "start_line": 1, "end_line": 1}]
        ),
    }


def test_pilot_returns_nonzero_when_checkout_fails(tmp_path, monkeypatch):
    from benchmarks import contextbench_pilot

    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([_task()]), encoding="utf-8")
    monkeypatch.setattr(
        contextbench_pilot,
        "_download",
        lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert contextbench_pilot.main(str(tasks), str(tmp_path / "checkouts"), 100, 1000) == 1


def test_floor_writes_explicitly_invalid_artifact_on_task_error(
    tmp_path,
    monkeypatch,
):
    from benchmarks import contextbench_floor, contextbench_pilot

    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([_task()]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        contextbench_pilot,
        "_download",
        lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert contextbench_floor.main(
        str(tasks),
        str(tmp_path / "checkouts"),
        100,
        1,
    ) == 1
    artifact = json.loads(
        (tmp_path / "benchmarks/results/contextbench_determinism_floor.json").read_text()
    )
    assert artifact["valid"] is False
    assert artifact["per_task"][0]["error"] == "RuntimeError: offline"


def test_subfile_writes_explicitly_invalid_artifact_on_task_error(
    tmp_path,
    monkeypatch,
):
    from benchmarks import contextbench_pilot, subfile_experiment

    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([_task()]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        contextbench_pilot,
        "_download",
        lambda *_: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert subfile_experiment.main(
        str(tasks),
        str(tmp_path / "checkouts"),
        1,
        100,
        2,
    ) == 1
    artifact = json.loads(
        (tmp_path / "benchmarks/results/subfile_experiment.json").read_text()
    )
    assert artifact["valid"] is False
    assert artifact["passed"] is False
    assert artifact["errors"][0]["error"] == "RuntimeError: offline"
