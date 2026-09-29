""""I could not check" and "I checked, it is neutral" are different answers.

`local_nli.nli_score` returned `("neutral", 0.5)` for both: a real neutral
verdict from the cross-encoder, and a model that never loaded at all. The
caller cannot tell them apart, so an unchecked claim is indistinguishable from
a checked one.

`relate.relation` already carries the honest vocabulary for this. A missing
backend reports `backend="unavailable"` with `uncertainty=1.0`. But a backend
that is present while its *model* never loads fell through to the neutral
branch and reported `backend="local_nli"` with `uncertainty=0.5` -- asserting
that an NLI model assessed the pair with moderate uncertainty when nothing had
been assessed. Measured offline, with the model absent from the cache, the load
attempt takes 10-64s and then every pair scores `("neutral", 0.5)`, including a
flat contradiction.

WITNESS routes both cases to its continuous risk path, so the final label does
not change. What changes is what the record says happened: it built an
`NLIVerdict("neutral", 0.5)` for a check that never ran.

These tests pin the distinction at all three layers, and pin that a genuine
neutral is still reported as one -- without that, the fix could make every
verdict read "unavailable" and still pass.
"""
from __future__ import annotations

import pytest

from entroly.relate.relation import score_relation
from entroly.verifiers import local_nli


@pytest.fixture(autouse=True)
def _reset_model_singleton():
    """The module caches its load attempt process-wide; isolate each test."""
    local_nli._pipeline = None
    local_nli._load_attempted = False
    local_nli._load_failed = False
    yield
    local_nli._pipeline = None
    local_nli._load_attempted = False
    local_nli._load_failed = False


class _StubPipeline:
    """Stands in for the cross-encoder, returning fixed logits."""

    def __init__(self, logits):
        self._logits = logits

    def predict(self, pairs, apply_softmax=False):  # noqa: ARG002
        return [self._logits for _ in pairs]


def _make_available(monkeypatch, logits):
    """Stub a *working* model, which needs the scoring stack to be importable.

    `nli_score` imports numpy and scipy.special inside its try block. Both
    arrive with the `neural` extra rather than the base install, so on a base
    wheel the import raises and every pair reports "unavailable" -- correctly,
    but it leaves nothing for these tests to stub. Skipping is right here:
    the base-install path is covered by
    `test_a_base_install_without_the_scoring_stack_reports_unavailable`.
    """
    # exc_type is explicit: pytest 9.1 turns the implicit ImportError handling
    # into an error, and this repo allows pytest<10.
    pytest.importorskip("numpy", reason="scoring stack ships with the neural extra",
                        exc_type=ImportError)
    pytest.importorskip("scipy", reason="scoring stack ships with the neural extra",
                        exc_type=ImportError)
    monkeypatch.setattr(local_nli, "_load_model", lambda: True)
    monkeypatch.setattr(local_nli, "_pipeline", _StubPipeline(logits), raising=False)


def _make_unavailable(monkeypatch):
    monkeypatch.setattr(local_nli, "_load_model", lambda: False)


# ── The scorer must say which of the two happened ────────────────────


def test_an_unloadable_model_is_reported_as_unavailable(monkeypatch):
    _make_unavailable(monkeypatch)

    label, _confidence = local_nli.nli_score("The sky is blue.", "The sky is green.")

    assert label == "unavailable", (
        "a model that never loaded reported a neutral verdict, which is a claim "
        "about the pair rather than about the check"
    )


def test_a_genuine_neutral_is_still_reported_as_neutral(monkeypatch):
    """Without this the fix could report everything unavailable and still pass."""
    # contradiction / entailment / neutral -- neutral wins, below both thresholds
    _make_available(monkeypatch, [0.0, 0.0, 5.0])

    label, confidence = local_nli.nli_score("Paris is in France.", "Paris has rain.")

    assert label == "neutral"
    assert confidence > 0.5


def test_a_real_verdict_is_unaffected(monkeypatch):
    """The scorer's actual job must keep working."""
    _make_available(monkeypatch, [8.0, 0.0, 0.0])

    label, confidence = local_nli.nli_score("The sky is blue.", "The sky is green.")

    assert label == "contradiction"
    assert confidence >= 0.65


def test_a_base_install_without_the_scoring_stack_reports_unavailable(monkeypatch):
    """The common case for `pip install entroly` with no extras.

    numpy and scipy arrive with the `neural` extra. Without them the import
    inside `nli_score` raises, and the honest answer is that no check ran --
    not a neutral verdict. This is the path CI's base-wheel job exercises, and
    it is why the stubbed tests above skip rather than fail there.
    """
    monkeypatch.setattr(local_nli, "_load_model", lambda: True)
    monkeypatch.setattr(local_nli, "_pipeline", _StubPipeline([0.0, 0.0, 5.0]),
                        raising=False)
    monkeypatch.setitem(__import__("sys").modules, "scipy", None)

    label, confidence = local_nli.nli_score("a", "b")

    assert label == "unavailable"
    assert confidence == 0.0


def test_batch_scoring_reports_unavailable_too(monkeypatch):
    """The batch path had the same conflation and the same callers."""
    _make_unavailable(monkeypatch)

    results = local_nli.batch_nli_scores("premise", ["a", "b"])

    assert [label for label, _ in results] == ["unavailable", "unavailable"]


# ── The relation layer must not stamp its own provenance on it ───────


def test_an_unavailable_check_is_not_attributed_to_the_nli_backend(monkeypatch):
    """`backend` is a provenance field; naming local_nli here is untrue."""
    _make_unavailable(monkeypatch)

    vector = score_relation("The sky is blue.", "The sky is green.",
                            backend=local_nli.nli_score)

    assert vector.backend == "unavailable"
    assert vector.uncertainty == 1.0, (
        "an unchecked pair reported partial certainty; the module already uses "
        "1.0 for the case where no backend exists at all"
    )


def test_an_unavailable_check_claims_no_support_or_contradiction(monkeypatch):
    _make_unavailable(monkeypatch)

    vector = score_relation("a", "b", backend=local_nli.nli_score)

    assert vector.support == 0.0
    assert vector.contradiction == 0.0


def test_a_genuine_neutral_is_still_attributed_to_the_backend(monkeypatch):
    """The honest case must keep its provenance, or the field means nothing."""
    _make_available(monkeypatch, [0.0, 0.0, 5.0])

    vector = score_relation("Paris is in France.", "Paris has rain.",
                            backend=local_nli.nli_score)

    assert vector.backend == "local_nli"
    assert vector.uncertainty < 1.0


# ── WITNESS must not mistake "unchecked" for "definitively checked" ───


def test_an_unavailable_nli_still_reaches_the_continuous_risk_model(monkeypatch):
    """This is the regression naming the new label would otherwise have caused.

    `use_continuous = nli is None or nli.label == "neutral"` (witness.py). While
    an unloadable model reported "neutral", an unchecked claim routed to the
    continuous risk model. Naming the label without handling it in witness.py
    makes that expression false and sends the claim down the discrete-bucket
    path -- which the code reserves for *definitive* entailment/contradiction
    verdicts. An unchecked claim would be treated as a confidently decided one.

    Asserting on proof-step names cannot detect this: an "unavailable" verdict
    adds no `nli_` step under either behaviour, so such a test passes with the
    fix reverted. The discriminator has to be whether the risk model ran, since
    only the continuous path consults it.
    """
    import entroly.witness as witness_mod

    _make_unavailable(monkeypatch)
    consulted: list[bool] = []
    real_get_model = witness_mod._get_default_risk_model

    def _spy():
        consulted.append(True)
        return real_get_model()

    monkeypatch.setattr(witness_mod, "_get_default_risk_model", _spy)
    # force_python: the Rust path bypasses the local NLI branch entirely.
    analyzer = witness_mod.WitnessAnalyzer(use_local_nli=True, force_python=True)

    result = analyzer.analyze("The sky is blue.", "The sky is green.")

    assert result.certificates, "no claims were certified; the test proves nothing"
    assert consulted, (
        "an unchecked claim skipped the continuous risk model and was routed "
        "through the path reserved for definitive NLI verdicts"
    )
