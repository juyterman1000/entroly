"""A persisted index must not be served to a different repository.

The warm-start index at ``<checkpoint_dir>/index.json.gz`` carried no corpus
identity: 32 keys, none naming the root it was built from. ``_ensure_index_loaded``
restored whatever the state directory happened to hold, so an engine that had
indexed *nothing* answered from another project's source. Measured before the
fix, with two repositories sharing one ``ENTROLY_DIR``::

    engine that indexed NOTHING returned: 1 fragments
       sources: ['file:src/secret_alpha.py']
       LEAKS repo A content: True
       any guidance/warning: []

That is a confidentiality failure, not a cache miss, and it triggers on the
ordinary setup of exporting ``ENTROLY_DIR`` globally or pinning it in a host
config so every project shares one state directory.

These tests pin the four states the check must distinguish: the root matches,
the root differs, provenance is unknown, and provenance could not be read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly.engine import EntrolyEngine


def _engine(state_dir: Path, source_root: Path, monkeypatch) -> EntrolyEngine:
    monkeypatch.setenv("ENTROLY_DIR", str(state_dir))
    monkeypatch.setenv("ENTROLY_SOURCE", str(source_root))
    return EntrolyEngine()


@pytest.fixture
def world(tmp_path, monkeypatch):
    state = tmp_path / "shared_state"
    (state).mkdir()
    a = tmp_path / "repoA"
    b = tmp_path / "repoB"
    for root in (a, b):
        (root / "src").mkdir(parents=True)
    (a / "src" / "alpha.py").write_text("ALPHA_ONLY = 1\n", encoding="utf-8")
    (b / "src" / "beta.py").write_text("BETA_ONLY = 2\n", encoding="utf-8")
    return state, a, b


def test_an_index_from_another_root_is_refused(world, monkeypatch) -> None:
    """The defect itself: a stranger's index must not be restored."""
    state, repo_a, repo_b = world
    engine = _engine(state, repo_a, monkeypatch)
    engine._stamp_index_owner(str(repo_a))

    serving_b = _engine(state, repo_b, monkeypatch)
    assert serving_b._warm_start_is_permitted() is False, (
        "an index built from another repository was accepted; this is how one "
        "project's source reached another project's agent"
    )


def test_the_owning_root_still_warm_starts(world, monkeypatch) -> None:
    """The fix must not disable warm start, only misdirected warm start."""
    state, repo_a, _repo_b = world
    engine = _engine(state, repo_a, monkeypatch)
    engine._stamp_index_owner(str(repo_a))

    again = _engine(state, repo_a, monkeypatch)
    assert again._warm_start_is_permitted() is True


def test_unknown_provenance_is_allowed_once_then_recorded(world, monkeypatch) -> None:
    """Pre-existing indexes carry no stamp; refusing them all would wipe caches.

    Allowed through, but it must not stay unknown -- the next session has to be
    able to decide properly.
    """
    state, repo_a, _repo_b = world
    engine = _engine(state, repo_a, monkeypatch)
    assert engine._recorded_index_root() == (None, "absent")
    assert engine._warm_start_is_permitted() is True
    recorded, basis = engine._recorded_index_root()
    assert basis == "stamp" and Path(recorded) == repo_a


def test_the_freshness_ledger_identifies_an_unstamped_index(world, monkeypatch) -> None:
    """An index written by a current build is identifiable without a stamp."""
    state, repo_a, repo_b = world
    (state / "index_freshness.json").write_text(
        json.dumps({"project_dir": str(repo_a)}), encoding="utf-8"
    )
    serving_b = _engine(state, repo_b, monkeypatch)
    recorded, basis = serving_b._recorded_index_root()
    assert basis == "freshness" and Path(recorded) == repo_a
    assert serving_b._warm_start_is_permitted() is False


def test_unreadable_provenance_is_not_reported_as_a_match(world, monkeypatch) -> None:
    """"Could not check" must never render as "it matches"."""
    state, repo_a, _repo_b = world
    (state / "index_owner.json").write_text("{not json", encoding="utf-8")
    engine = _engine(state, repo_a, monkeypatch)
    recorded, basis = engine._recorded_index_root()
    assert recorded is None
    assert basis == "unreadable", "a corrupt stamp must be distinguishable from an absent one"


def test_a_refusal_does_not_retry_on_every_call(world, monkeypatch) -> None:
    """A refusal marks the engine loaded-and-empty, not permanently retrying."""
    state, repo_a, repo_b = world
    _engine(state, repo_a, monkeypatch)._stamp_index_owner(str(repo_a))
    serving_b = _engine(state, repo_b, monkeypatch)
    serving_b._ensure_index_loaded()
    assert serving_b._index_loaded is True


def test_an_engine_that_indexed_nothing_serves_nothing(tmp_path, monkeypatch) -> None:
    """The end-to-end leak, asserted without referencing any new symbol.

    This is the test that matters: it fails against the pre-fix engine on
    BEHAVIOUR, not on a missing attribute. Repo A indexes and persists; a
    second engine for repo B opens the same state directory and indexes
    nothing. Before the fix it was handed repo A's source.
    """
    from entroly.auto_index import auto_index

    state = tmp_path / "shared"
    state.mkdir()
    repo_a = tmp_path / "projA"
    repo_b = tmp_path / "projB"
    (repo_a / "src").mkdir(parents=True)
    (repo_b / "src").mkdir(parents=True)
    (repo_a / "src" / "alpha.py").write_text(
        "def ALPHA_ONLY_MARKER_payments(order):\n    return order\n", encoding="utf-8"
    )
    (repo_b / "src" / "beta.py").write_text(
        "def BETA_ONLY_MARKER_sessions(s):\n    return s\n", encoding="utf-8"
    )

    monkeypatch.setenv("ENTROLY_DIR", str(state))
    monkeypatch.setenv("ENTROLY_SOURCE", str(repo_a))
    owner = EntrolyEngine()
    auto_index(owner, str(repo_a))
    try:
        owner.persist_index()
    except Exception:  # noqa: BLE001 - persistence may be config-gated
        pass

    monkeypatch.setenv("ENTROLY_SOURCE", str(repo_b))
    intruder = EntrolyEngine()
    result = intruder.optimize_context(token_budget=800, query="payments marker")
    served = " ".join(
        str(f.get("content", "")) for f in (result.get("selected_fragments") or [])
    )
    assert "ALPHA_ONLY_MARKER" not in served, (
        "an engine serving projB was handed projA's source from the shared "
        "state directory"
    )
