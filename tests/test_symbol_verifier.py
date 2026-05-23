"""
Tests for entroly.verifiers.symbol_resolution
==============================================

This is the proof that the Bayesian hallucination verifier:
  1. Passes legitimate code (low false-positive rate)
  2. Catches the four hallucination classes:
     - Invented imports (nonexistent top-level packages)
     - Invented methods on real modules (e.g., requests.fakeMethod())
     - Plausible-but-typo'd names (compress_msgs vs compress_messages)
     - Fully fabricated identifiers (LLM-style invention)
  3. Is consistent: same input → same H score (deterministic)
  4. Respects the math: H ∈ [0,1], unresolved → H = 1
"""

from __future__ import annotations

import math
import pytest

from entroly.verifiers import (
    CharNGramModel,
    SymbolManifest,
    SymbolVerifier,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def small_manifest() -> SymbolManifest:
    """A minimal manifest with known good symbols for unit tests."""
    return SymbolManifest(
        repo={"compress", "compress_messages", "MyClass", "fetch_user"},
        stdlib={"json", "os", "sys", "re", "math", "dumps", "loads", "path"},
        installed={"numpy", "requests", "torch", "pandas", "entroly"},
        builtins={"print", "len", "dict", "list", "True", "False", "None"},
    )


@pytest.fixture
def trained_ngram() -> CharNGramModel:
    """N-gram trained on Python-flavored code."""
    m = CharNGramModel(n=4)
    m.train_from_strings([
        "def authenticate(user): pass",
        "def fetch_user(user_id): return db.users.get(user_id)",
        "class MyClass: pass",
        "from entroly import compress, compress_messages",
        "import json; data = json.loads(text)",
        "def process_request(req): return handler(req)",
    ] * 50)  # repeat for more stable statistics
    return m


@pytest.fixture
def verifier(small_manifest, trained_ngram) -> SymbolVerifier:
    return SymbolVerifier(
        manifest=small_manifest,
        ngram_model=trained_ngram,
        lambda_calibration=6.5,
    )


# ── Manifest semantics ───────────────────────────────────────────────


class TestSymbolManifest:
    def test_contains_in_any_layer(self, small_manifest):
        assert "compress" in small_manifest        # repo
        assert "json" in small_manifest             # stdlib
        assert "requests" in small_manifest         # installed
        assert "print" in small_manifest            # builtins
        assert "nonexistent_xyz" not in small_manifest

    def test_provenance_reports_layer(self, small_manifest):
        assert small_manifest.provenance("compress") == "repo"
        assert small_manifest.provenance("json") == "stdlib"
        assert small_manifest.provenance("requests") == "installed"
        assert small_manifest.provenance("print") == "builtins"
        assert small_manifest.provenance("nope") is None

    def test_size_is_sum(self, small_manifest):
        assert small_manifest.size() == (
            len(small_manifest.repo)
            + len(small_manifest.stdlib)
            + len(small_manifest.installed)
            + len(small_manifest.builtins)
        )

    def test_imports_are_not_added_to_manifest(self, tmp_path):
        """The critical bug-class: imports must NOT contribute to manifest.

        Otherwise hallucinated imports would self-validate.
        """
        src = tmp_path / "mod.py"
        src.write_text(
            "from torch.nn import HyperbolicAttention\n"
            "from fake_lib import FakeClass\n"
            "def real_func(): pass\n"
            "class RealClass: pass\n"
        )
        manifest = SymbolManifest.build_from_codebase(str(tmp_path))
        # Defined locally → in manifest
        assert "real_func" in manifest.repo
        assert "RealClass" in manifest.repo
        # Imported from elsewhere → MUST NOT be in repo manifest
        assert "HyperbolicAttention" not in manifest.repo
        assert "FakeClass" not in manifest.repo


# ── N-gram model ─────────────────────────────────────────────────────


class TestCharNGramModel:
    def test_known_words_have_low_surprisal(self, trained_ngram):
        """Words from training corpus should be unsurprising."""
        for word in ["authenticate", "fetch_user", "compress", "request"]:
            s = trained_ngram.surprisal(word)
            assert s < 3.0, f"{word!r} surprisal {s:.2f} unexpectedly high"

    def test_random_strings_have_high_surprisal(self, trained_ngram):
        """Random/unusual character sequences should be very surprising."""
        for word in ["xqzybw", "qqqqqqqq", "zxcvbnmasdfgh"]:
            s = trained_ngram.surprisal(word)
            assert s > 5.0, f"{word!r} surprisal {s:.2f} unexpectedly low"

    def test_surprisal_is_nonnegative(self, trained_ngram):
        for word in ["abc", "", "xyz", "compress"]:
            assert trained_ngram.surprisal(word) >= 0

    def test_empty_string_returns_zero(self, trained_ngram):
        assert trained_ngram.surprisal("") == 0.0

    def test_pickling_roundtrip(self, trained_ngram):
        """Cache layer relies on pickling — must round-trip cleanly."""
        import pickle
        blob = pickle.dumps(trained_ngram)
        recovered = pickle.loads(blob)
        for word in ["authenticate", "xyzqrt", "compress"]:
            assert math.isclose(
                trained_ngram.surprisal(word),
                recovered.surprisal(word),
                rel_tol=1e-9,
            )


# ── The Bayesian posterior ───────────────────────────────────────────


class TestPosteriorMath:
    def test_unresolved_symbol_is_definitely_hallucinated(self, verifier):
        """If σ ∉ M, P(θ=0|σ) = 1 by construction."""
        from entroly.verifiers.symbol_resolution import SymbolReference
        ref = SymbolReference(
            name="totally_made_up_xyzzy",
            kind="call", line=1, weight=1.0,
        )
        assert verifier.posterior_hallucinated(ref) == 1.0

    def test_resolved_low_surprisal_is_grounded(self, verifier):
        from entroly.verifiers.symbol_resolution import SymbolReference
        ref = SymbolReference(name="compress", kind="call", line=1, weight=1.0)
        p = verifier.posterior_hallucinated(ref)
        assert p < 0.1, f"P_halu={p} for legitimate 'compress' too high"

    def test_resolved_high_surprisal_is_suspicious(self, verifier):
        """Symbol in manifest BUT unusual character distribution → suspicious."""
        # Add a high-surprisal name to the manifest to force the mixed case
        verifier.manifest.repo.add("zxqyvtbprwklm")
        from entroly.verifiers.symbol_resolution import SymbolReference
        ref = SymbolReference(
            name="zxqyvtbprwklm", kind="call", line=1, weight=1.0,
        )
        p = verifier.posterior_hallucinated(ref)
        assert p > 0.3, f"P_halu={p} for high-surprisal symbol too low"

    def test_dunder_methods_always_pass(self, verifier):
        from entroly.verifiers.symbol_resolution import SymbolReference
        for name in ["__init__", "__str__", "__repr__", "__getattr__"]:
            ref = SymbolReference(name=name, kind="call", line=1, weight=1.0)
            assert verifier.posterior_hallucinated(ref) == 0.0

    def test_h_score_is_in_unit_interval(self, verifier):
        """H ∈ [0, 1] regardless of input."""
        for code in [
            "x = 1",
            "import os",
            "fakeFunc(); otherFakeFunc(); moreFakes()",
            "from compress import compress; compress('x')",
        ]:
            r = verifier.verify(code)
            assert 0.0 <= r.h_score <= 1.0


# ── End-to-end hallucination detection ───────────────────────────────


class TestHallucinationDetection:
    def test_legitimate_code_passes(self, verifier):
        code = """
from entroly import compress, compress_messages
def helper():
    return compress("x", budget=100)
"""
        r = verifier.verify(code)
        assert r.passed(), f"legit code failed: H={r.h_score}, unresolved={r.unresolved_symbols()}"

    def test_invented_import_fails(self, verifier):
        code = """
from quantumtorch.advanced import HyperbolicSVM
model = HyperbolicSVM()
"""
        r = verifier.verify(code)
        assert not r.passed()
        unresolved = r.unresolved_symbols()
        assert "quantumtorch" in unresolved
        assert "HyperbolicSVM" in unresolved

    def test_invented_method_on_real_module_fails(self, verifier):
        code = """
import requests
r = requests.fetchAuthSessionFromBlockchain()
"""
        r = verifier.verify(code)
        assert not r.passed()
        assert "fetchAuthSessionFromBlockchain" in r.unresolved_symbols()

    def test_plausible_typo_fails(self, verifier):
        """compress_msgs is a plausible typo for compress_messages — must catch."""
        code = """
from entroly import compress_msgs
result = compress_msgs([])
"""
        r = verifier.verify(code)
        assert not r.passed()
        assert "compress_msgs" in r.unresolved_symbols()

    def test_fully_fabricated_llm_output(self, verifier):
        """Classic LLM-style hallucination of multiple fake APIs."""
        code = """
from sklearn.advanced_kernels import HyperbolicSVM
from torch.experimental import quantum_adam
model = HyperbolicSVM(dimensions=42)
optimizer = quantum_adam(model.parameters())
"""
        r = verifier.verify(code)
        assert not r.passed()
        # Each of these must be flagged
        for sym in ["HyperbolicSVM", "quantum_adam"]:
            assert sym in r.unresolved_symbols(), f"failed to flag {sym}"


# ── Determinism ──────────────────────────────────────────────────────


class TestDeterminism:
    def test_repeated_verify_gives_same_score(self, verifier):
        code = "from entroly import compress; compress('x')"
        r1 = verifier.verify(code)
        r2 = verifier.verify(code)
        assert math.isclose(r1.h_score, r2.h_score, rel_tol=1e-12)
        assert len(r1.judgments) == len(r2.judgments)


# ── Performance budget ───────────────────────────────────────────────


class TestPerformance:
    def test_verify_is_fast(self, verifier):
        """Inference path (after manifest+ngram are built) must be <100ms
        for a 100-line code blob. The slow part is one-time build."""
        import time
        code = "\n".join([
            "from entroly import compress",
            *[f"x{i} = compress(text='x', budget={i*10})" for i in range(50)]
        ])
        t0 = time.perf_counter()
        for _ in range(10):
            verifier.verify(code)
        avg_ms = (time.perf_counter() - t0) * 100  # /10 then *1000
        assert avg_ms < 100, f"verify avg {avg_ms:.1f}ms > 100ms budget"
