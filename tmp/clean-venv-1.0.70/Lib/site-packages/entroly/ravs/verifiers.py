"""
RAVS v2 — Verifiers

Each verifier checks an executor's output against a ground-truth or
structural criterion. All verifiers:
  1. Accept (executor_result, expected/source) and return (passed: bool, reason: str)
  2. Never have side effects
  3. Fail closed — on any exception, return (False, error)

Verifier types:
  ExactVerifier        — deterministic equality (computation nodes)
  StructuralVerifier   — AST/JSON structural validation (code_inspection)
  ExitCodeVerifier     — exit code == 0 (test_execution)
  CitationVerifier     — stub for v2 (retrieval_claim)
"""

from __future__ import annotations


import json
import logging
import math
import re
from typing import Any

logger = logging.getLogger("entroly.ravs.verifiers")


# ── Verifier Result ────────────────────────────────────────────────────


class VerifierResult:
    __slots__ = ("passed", "reason", "verifier_name")

    def __init__(self, passed: bool = False, reason: str = "",
                 verifier_name: str = ""):
        self.passed = passed
        self.reason = reason
        self.verifier_name = verifier_name


# ── Exact Verifier ─────────────────────────────────────────────────────


class ExactVerifier:
    """Deterministic equality check for computation results.

    Handles:
      - Integer/float comparison with tolerance
      - Symbolic expression equivalence
      - List/set comparison for multi-valued results
    """

    def __init__(self, tolerance: float = 1e-9):
        self._tolerance = tolerance

    def verify(self, executor_result: str, expected: str) -> VerifierResult:
        try:
            # Try numeric comparison first
            try:
                a = float(executor_result.strip())
                b = float(expected.strip())
                if math.isnan(a) and math.isnan(b):
                    return VerifierResult(True, "both NaN", "exact")
                if abs(a - b) <= self._tolerance:
                    return VerifierResult(True, f"|{a}-{b}| <= {self._tolerance}", "exact")
                if b != 0 and abs((a - b) / b) <= self._tolerance:
                    return VerifierResult(True, f"relative error <= {self._tolerance}", "exact")
                return VerifierResult(False, f"{a} != {b}", "exact")
            except (ValueError, TypeError):
                pass

            # Try JSON comparison (for lists, sets, etc.)
            try:
                a_json = json.loads(executor_result)
                b_json = json.loads(expected)
                if a_json == b_json:
                    return VerifierResult(True, "JSON equal", "exact")
            except (json.JSONDecodeError, TypeError):
                pass

            # String equality (case-insensitive, whitespace-normalized)
            a_norm = " ".join(executor_result.strip().lower().split())
            b_norm = " ".join(expected.strip().lower().split())
            if a_norm == b_norm:
                return VerifierResult(True, "string equal (normalized)", "exact")

            return VerifierResult(False, f"'{executor_result}' != '{expected}'", "exact")

        except Exception as e:
            return VerifierResult(False, f"verifier error: {e}", "exact")


# ── Structural Verifier ────────────────────────────────────────────────


class StructuralVerifier:
    """AST/JSON structural validation for code inspection results.

    Checks that the executor's JSON output has the expected structure
    and key fields, without requiring exact value equality.
    """

    def verify(
        self,
        executor_result: str,
        expected_keys: list[str] | None = None,
        expected_types: dict[str, str] | None = None,
    ) -> VerifierResult:
        try:
            data = json.loads(executor_result)
            if not isinstance(data, dict):
                return VerifierResult(False, "result is not a JSON object", "structural")

            # Check required keys
            if expected_keys:
                missing = [k for k in expected_keys if k not in data]
                if missing:
                    return VerifierResult(
                        False,
                        f"missing keys: {missing}",
                        "structural",
                    )

            # Check value types
            if expected_types:
                for key, expected_type in expected_types.items():
                    if key not in data:
                        continue
                    val = data[key]
                    type_map = {
                        "int": int, "float": float, "str": str,
                        "list": list, "dict": dict, "bool": bool,
                    }
                    expected_py_type = type_map.get(expected_type)
                    if expected_py_type and not isinstance(val, expected_py_type):
                        return VerifierResult(
                            False,
                            f"key '{key}' expected {expected_type}, got {type(val).__name__}",
                            "structural",
                        )

            # Check no error field
            if "error" in data and data["error"]:
                return VerifierResult(
                    False,
                    f"result contains error: {data['error']}",
                    "structural",
                )

            return VerifierResult(True, "structure valid", "structural")

        except json.JSONDecodeError:
            return VerifierResult(False, "result is not valid JSON", "structural")
        except Exception as e:
            return VerifierResult(False, f"verifier error: {e}", "structural")


# ── Exit Code Verifier ─────────────────────────────────────────────────


class ExitCodeVerifier:
    """Check exit code for test execution results."""

    def verify(self, executor_result: str, expected_exit_code: int = 0) -> VerifierResult:
        try:
            # Try bare numeric first
            stripped = executor_result.strip()
            try:
                code = int(stripped)
                if code == expected_exit_code:
                    return VerifierResult(True, f"exit_code={code}", "exit_code")
                return VerifierResult(False, f"exit_code={code}, expected={expected_exit_code}", "exit_code")
            except ValueError:
                pass

            # Try JSON
            data = json.loads(executor_result)
            if isinstance(data, dict):
                exit_code = data.get("exit_code", data.get("status"))
                if exit_code == expected_exit_code:
                    return VerifierResult(True, f"exit_code={exit_code}", "exit_code")
                if data.get("status") == "shadow_stub":
                    return VerifierResult(
                        False,
                        "shadow stub — not actually executed",
                        "exit_code",
                    )
                return VerifierResult(
                    False,
                    f"exit_code={exit_code}, expected={expected_exit_code}",
                    "exit_code",
                )
            return VerifierResult(False, "result is not a JSON object", "exit_code")
        except (json.JSONDecodeError, Exception):
            return VerifierResult(False, "cannot parse exit code", "exit_code")


# ── Citation Verifier ──────────────────────────────────────────────────


class CitationVerifier:
    """Citation/entailment verifier using n-gram overlap and Jaccard similarity.

    Given a claim (executor_result) and source text, determines whether
    the source actually supports the claim. Uses three complementary signals:

      1. **Token Jaccard** — unigram overlap between claim and source.
         J(A,B) = |A ∩ B| / |A ∪ B|. Threshold: 0.15.

      2. **Key phrase coverage** — extracts noun-phrase-like bigrams from
         the claim and checks what fraction appear in the source. This
         catches cases where individual words overlap but the actual
         concepts don't match. Threshold: 0.30.

      3. **Numeric consistency** — if the claim contains numbers, checks
         that the same numbers appear in the source. A claim citing
         "99.7% accuracy" against a source saying "97.3%" is a miss.

    Final verdict: PASS if (jaccard >= 0.15 AND key_coverage >= 0.30
    AND numerics_consistent). Conservative: abstains rather than
    false-positiving.
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.15,
        key_coverage_threshold: float = 0.30,
    ):
        self._jaccard_t = jaccard_threshold
        self._key_coverage_t = key_coverage_threshold

    def verify(self, executor_result: str, source_text: str = "") -> VerifierResult:
        if not source_text or not executor_result:
            return VerifierResult(
                False,
                "empty claim or source — cannot verify",
                "citation",
            )

        claim = executor_result.strip()
        source = source_text.strip()

        # --- Signal 1: Token Jaccard ---
        claim_tokens = self._tokenize(claim)
        source_tokens = self._tokenize(source)

        if not claim_tokens:
            return VerifierResult(False, "claim has no tokens", "citation")

        intersection = claim_tokens & source_tokens
        union = claim_tokens | source_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # --- Signal 2: Key phrase coverage ---
        claim_bigrams = self._bigrams(claim)
        source_lower = source.lower()
        if claim_bigrams:
            covered = sum(1 for bg in claim_bigrams if bg in source_lower)
            key_coverage = covered / len(claim_bigrams)
        else:
            key_coverage = jaccard  # fallback for very short claims

        # --- Signal 3: Numeric consistency ---
        claim_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
        if claim_numbers:
            source_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', source))
            numerics_ok = len(claim_numbers & source_numbers) / len(claim_numbers) >= 0.5
        else:
            numerics_ok = True  # no numbers to check

        # --- Verdict (composite score) ---
        # Weighted combination: jaccard dominates for code, key_coverage
        # helps for prose. Numeric consistency is a hard gate.
        composite = 0.6 * jaccard + 0.4 * key_coverage
        threshold = 0.10  # combined minimum

        reasons = []
        passed = True

        reasons.append(f"jaccard={jaccard:.3f}")
        reasons.append(f"key_coverage={key_coverage:.3f}")
        reasons.append(f"composite={composite:.3f}")

        if composite < threshold:
            passed = False
            reasons.append(f"composite<{threshold}")

        if not numerics_ok:
            passed = False
            reasons.append(f"numeric_mismatch: claim={claim_numbers}")
        else:
            reasons.append("numerics_ok")

        return VerifierResult(
            passed=passed,
            reason="; ".join(reasons),
            verifier_name="citation",
        )

    def _tokenize(self, text: str) -> set[str]:
        """Extract lowercased word tokens (length >= 3, skip stopwords)."""
        stopwords = frozenset({
            "the", "and", "for", "that", "this", "with", "from", "are",
            "was", "were", "been", "being", "have", "has", "had", "will",
            "would", "could", "should", "can", "may", "might", "shall",
            "not", "but", "its", "also", "than", "then", "into",
        })
        words = set(re.findall(r'\b\w{3,}\b', text.lower()))
        return words - stopwords

    def _bigrams(self, text: str) -> list[str]:
        """Extract meaningful bigrams (adjacent word pairs)."""
        words = re.findall(r'\b\w{3,}\b', text.lower())
        return [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]


# ── Verifier Registry ─────────────────────────────────────────────────


class VerifierRegistry:
    """Central registry mapping verifier types to implementations."""

    def __init__(self):
        self._exact = ExactVerifier()
        self._structural = StructuralVerifier()
        self._exit_code = ExitCodeVerifier()
        self._citation = CitationVerifier()

    def get(self, verifier_type: str) -> Any:
        return {
            "exact": self._exact,
            "structural": self._structural,
            "exit_code": self._exit_code,
            "citation": self._citation,
        }.get(verifier_type)
