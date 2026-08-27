from __future__ import annotations

import math

import pytest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from entroly.verifiers.scope_analyzer import ReverseIndex
from entroly.verifiers.service import VerifierService, _ServiceInstance
from entroly.verifiers.symbol_resolution import (
    MANIFEST_EVIDENCE_LOG_ODDS,
    SymbolManifest,
    SymbolVerifier,
)


def _expected_posterior(surprisal: float, lambda_used: float) -> float:
    """The posterior this configuration should produce, derived not guessed.

    Pinning the formula rather than a literal keeps the test meaningful when
    the calibration constants move: it still fails if per-archetype lambda
    stops being applied, which is what this test exists to guard.
    """
    x = surprisal - lambda_used + MANIFEST_EVIDENCE_LOG_ODDS
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class _FixedSurprisal:
    def surprisal(self, _name: str) -> float:
        return 7.0


def test_cached_service_strict_policy_is_monotonic(tmp_path) -> None:
    VerifierService.shutdown_all()
    try:
        permissive = VerifierService.for_repo(str(tmp_path), run_type_check=False)
        strict = VerifierService.for_repo(str(tmp_path), run_type_check=True)

        assert strict is permissive
        assert strict._run_type_check is True

        # A later permissive caller must not silently downgrade the shared service.
        again = VerifierService.for_repo(str(tmp_path), run_type_check=False)
        assert again is strict
        assert again._run_type_check is True
    finally:
        VerifierService.shutdown_all()


def test_concurrent_archetypes_do_not_mutate_shared_lambda(
    monkeypatch, tmp_path
) -> None:
    instance = _ServiceInstance(str(tmp_path), run_type_check=False)
    base_verifier = SymbolVerifier(
        manifest=SymbolManifest(repo={"foo"}),
        ngram_model=_FixedSurprisal(),
        lambda_calibration=42.0,
    )
    reverse_index = ReverseIndex()

    monkeypatch.setattr(
        instance,
        "_snapshot_ready_state",
        lambda: (base_verifier, reverse_index, False),
    )
    instance._calibrator = SimpleNamespace(
        get=lambda archetype: 0.0 if archetype == "strict" else 100.0
    )

    def run(archetype: str) -> tuple[str, float, float]:
        result = instance.verify("foo()", archetype=archetype)
        probability = result.judgments[0].base.p_hallucinated
        return archetype, result.lambda_used, probability

    archetypes = ["strict", "permissive"] * 100
    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(run, archetypes))

    assert base_verifier.lambda_ == 42.0
    strict_seen: list[float] = []
    permissive_seen: list[float] = []
    for archetype, lambda_used, probability in outcomes:
        if archetype == "strict":
            assert lambda_used == 0.0
            assert probability == pytest.approx(
                _expected_posterior(7.0, 0.0), rel=1e-9
            )
            strict_seen.append(probability)
        else:
            assert lambda_used == 100.0
            assert probability < 0.01
            permissive_seen.append(probability)

    # The point of the two archetypes is that lambda actually reaches the
    # posterior. Assert the separation directly rather than trusting a literal
    # threshold, which previously encoded the pre-correction absolute value.
    assert min(strict_seen) > max(permissive_seen) * 1000


def test_snapshot_survives_invalidation_without_none_state(
    monkeypatch, tmp_path
) -> None:
    instance = _ServiceInstance(str(tmp_path), run_type_check=False)
    base_verifier = SymbolVerifier(
        manifest=SymbolManifest(repo={"foo"}),
        ngram_model=None,
        lambda_calibration=1.0,
    )
    reverse_index = ReverseIndex()

    monkeypatch.setattr(
        instance,
        "_ensure_ready",
        lambda force_rebuild=False: None,
    )
    instance._verifier = base_verifier
    instance._reverse_index = reverse_index

    verifier, index, default_type_check = instance._snapshot_ready_state()
    instance.invalidate()

    assert verifier is base_verifier
    assert index is reverse_index
    assert default_type_check is False
    assert instance._verifier is None
    assert instance._reverse_index is None
