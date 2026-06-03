import pytest

entroly_core = pytest.importorskip("entroly_core", reason="Rust path not installed")
EntrolyEngine = entroly_core.EntrolyEngine


def _engine():
    return EntrolyEngine(
        w_recency=0.30,
        w_frequency=0.25,
        w_semantic=0.25,
        w_entropy=0.20,
        decay_half_life=15,
        min_relevance=0.05,
        hamming_threshold=3,
        exploration_rate=0.0,
        max_fragments=100,
        enable_ios=True,
        enable_ios_diversity=True,
        enable_ios_multi_resolution=True,
    )


def test_recovery_pushes_skeleton_resolution_toward_full_context():
    engine = _engine()
    before = engine.get_skeleton_info_factor()

    engine.record_resolution_outcome(["skeleton"], False)

    assert engine.get_skeleton_info_factor() < before


def test_verified_compression_recovers_slowly_after_a_failure():
    engine = _engine()
    baseline = engine.get_reference_info_factor()

    engine.record_resolution_outcome(["reference"], False)
    after_failure = engine.get_reference_info_factor()
    engine.record_resolution_outcome(["reference"], True)
    after_pass = engine.get_reference_info_factor()

    assert after_failure < baseline
    assert after_failure < after_pass < baseline


def test_duplicate_resolution_labels_apply_only_one_update():
    engine = _engine()
    before = engine.get_skeleton_info_factor()

    engine.record_resolution_outcome(["skeleton", "skeleton"], False)

    assert engine.get_skeleton_info_factor() == before - 0.040
