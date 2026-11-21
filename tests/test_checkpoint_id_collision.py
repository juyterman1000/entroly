"""Two checkpoints must never share a filename.

The id was ``ckpt_{instance_id}_{unix_seconds}_{per_manager_counter}``, and
``instance_id`` is host+pid. Two ``CheckpointManager`` objects in ONE process
therefore shared the instance id *and* each began its own counter at zero, so
two saves in the same second minted one filename. Measured before the fix::

    A wrote: ckpt_da333bd32ca7_29084_1795245521_0.json.gz
    B wrote: ckpt_da333bd32ca7_29084_1795245521_0.json.gz
    SAME FILE (collision): True
    files after 2 saves: 1
    surviving fragments: ['BBB_ONLY']
    A's checkpoint survived: False

Two saves, one file. The second overwrote the first with no exception, no
warning, and no hint in the returned path -- silent data loss of exactly the
state a checkpoint exists to protect.

The id shape is load-bearing: ``_AUTO_ID_RE``, ``_peer_pid``, the
per-instance globs and the peer-reaping scan all read ``ckpt_{host}_{pid}_``.
These tests pin uniqueness AND that those readers still work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entroly.checkpoint import CheckpointManager, ContextFragment


def _fragment(fragment_id: str, body: str) -> ContextFragment:
    return ContextFragment(fragment_id, body, 5, f"{fragment_id}.py")


def _save(manager: CheckpointManager, fragment_id: str) -> Path:
    return Path(
        manager.save(
            [_fragment(fragment_id, f"body of {fragment_id}")],
            {}, {}, 1, metadata={"who": fragment_id},
        )
    )


def _fragment_ids(checkpoint) -> list[str]:
    records = getattr(checkpoint, "fragments", None) or []
    out = []
    for record in records:
        if isinstance(record, dict):
            out.append(record.get("fragment_id") or record.get("id"))
        else:
            out.append(getattr(record, "fragment_id", None))
    return [value for value in out if value]


def test_two_managers_in_one_process_do_not_overwrite_each_other(tmp_path) -> None:
    """The defect: same host, same pid, both counters at zero."""
    first = CheckpointManager(checkpoint_dir=str(tmp_path))
    second = CheckpointManager(checkpoint_dir=str(tmp_path))
    # The shared instance id is the precondition; the fix does not remove it.
    assert first.instance_id == second.instance_id

    path_a = _save(first, "AAA_ONLY")
    path_b = _save(second, "BBB_ONLY")

    assert path_a.name != path_b.name, "two managers minted the same filename"
    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) == 2, (
        "two saves produced fewer than two files; one overwrote the other"
    )


def test_both_checkpoints_remain_loadable(tmp_path) -> None:
    """Uniqueness is only worth having if the data survives."""
    first = CheckpointManager(checkpoint_dir=str(tmp_path))
    second = CheckpointManager(checkpoint_dir=str(tmp_path))
    path_a = _save(first, "AAA_ONLY")
    path_b = _save(second, "BBB_ONLY")

    assert _fragment_ids(first.load_by_id(path_a.name[: -len(".json.gz")])) == ["AAA_ONLY"]
    assert _fragment_ids(second.load_by_id(path_b.name[: -len(".json.gz")])) == ["BBB_ONLY"]


def test_many_saves_in_the_same_second_are_all_distinct(tmp_path) -> None:
    """The collision window is one second wide, so burst within it."""
    managers = [CheckpointManager(checkpoint_dir=str(tmp_path)) for _ in range(4)]
    names = {_save(m, f"F{i}").name for i, m in enumerate(managers) for _ in (0,)}
    assert len(names) == 4, f"expected 4 distinct checkpoints, got {sorted(names)}"
    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) == 4


def test_an_existing_file_is_never_clobbered(tmp_path) -> None:
    """Covers causes a counter cannot: reused pid, explicit id, clock step."""
    manager = CheckpointManager(checkpoint_dir=str(tmp_path))
    taken = manager._next_checkpoint_id()
    (tmp_path / f"{taken}.json.gz").write_bytes(b"pre-existing")

    # Force the same base id by rewinding the process-wide sequence.
    CheckpointManager._id_sequence -= 1
    nxt = manager._next_checkpoint_id()
    assert nxt != taken, "minted an id whose file already exists"
    assert (tmp_path / f"{taken}.json.gz").read_bytes() == b"pre-existing"


@pytest.mark.parametrize("make_second", [True, False])
def test_the_id_shape_readers_depend_on_still_parses(tmp_path, make_second) -> None:
    """`_AUTO_ID_RE`, `_peer_pid` and the per-instance glob read the prefix."""
    manager = CheckpointManager(checkpoint_dir=str(tmp_path))
    if make_second:
        CheckpointManager(checkpoint_dir=str(tmp_path))
    name = _save(manager, "AAA_ONLY").name

    owner = CheckpointManager._checkpoint_owner(name)
    assert owner is not None, f"{name!r} no longer parses as an auto checkpoint"
    instance_id, _host, pid = owner
    assert instance_id == manager.instance_id
    assert pid > 0
    assert manager._peer_pid(name) == pid
    assert list(tmp_path.glob(f"ckpt_{manager.instance_id}_*.json.gz"))
