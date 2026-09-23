"""Conditional compression residual: a model-free redundancy diagnostic.

Entroly measures how many additional compressed bytes a fragment contributes
after a retained prefix. The measurement is deterministic for a fixed runtime
and codec configuration, and is treated only as a replayable heuristic:

    K(F | R)  ~=  C(R + F) - C(R)

Normalising by F's own compressed size gives the **conditional compression
residual**:

    CCR(F | R) = [C(R + F) - C(R)] / C(F)

    CCR -> 0   Appending F changes compressed size little.
    CCR -> 1   Appending F costs roughly its standalone compressed size.

Why this and not embedding cosine similarity:

1. **Asymmetry.**  Omission safety is directional.  A short fragment may be
   fully recoverable from a longer one while the reverse is false.  Cosine
   similarity is symmetric and cannot represent that distinction at all;
   this is a representational limit, not a tuning limit.
2. **No model download.**  ``zlib``, ``bz2`` and ``lzma`` are standard
   library, preserving local-first operation.
3. **Determinism.**  Repeatable with fixed codec implementations and settings.
   Embedding scores drift across model versions, which breaks receipt
   replay.
4. **Explainability.**  "Appending F to R cost 12 of 87 bytes" is an
   inspectable receipt line; a cosine of 0.83 is not.

Ensemble rule: compressors have different inductive biases (LZ77 literal
repeats, BWT reordering, LZMA long-range structure).  This implementation takes the maximum
residual across compressors. This prevents a permissive codec from deciding
alone but does not turn agreement into proof of omission safety.

This module measures byte redundancy only; it does not prove recoverability.  It cannot answer *constraint
preservation*: "deployment requires security approval" and "deployment
requires performance review" compress well against each other yet impose
different obligations.  Structural checks remain mandatory; see
``entroly.relate.info_residual``.
"""
from __future__ import annotations

import bz2
import lzma
import zlib
from dataclasses import dataclass
from typing import Callable

_SEP = b"\n"
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 6}]


def _zlib_size(data: bytes) -> int:
    return len(zlib.compress(data, 9))


def _bz2_size(data: bytes) -> int:
    return len(bz2.compress(data, 9))


def _lzma_size(data: bytes) -> int:
    return len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))


COMPRESSORS: dict[str, Callable[[bytes], int]] = {
    "zlib": _zlib_size,
    "bz2": _bz2_size,
    "lzma": _lzma_size,
}

_EMPTY_OVERHEAD = {name: fn(b"") for name, fn in COMPRESSORS.items()}


def _adjusted(name: str, data: bytes) -> int:
    """Compressed size with the codec's fixed header cost removed."""
    return max(1, COMPRESSORS[name](data) - _EMPTY_OVERHEAD[name])


@dataclass(frozen=True)
class CompressionCertificate:
    """Replayable compression diagnostic.

    Every field needed to recompute the verdict is recorded, so an auditor
    can reproduce it given the fragment, retained bytes and codec environment.
    """

    residual: float
    per_compressor: tuple[tuple[str, float], ...]
    deciding_compressor: str
    threshold: float
    recoverable: bool
    retained_bytes: int
    fragment_bytes: int

    def to_dict(self) -> dict:
        return {
            "residual": round(self.residual, 4),
            "per_compressor": {k: round(v, 4) for k, v in self.per_compressor},
            "deciding_compressor": self.deciding_compressor,
            "threshold": self.threshold,
            "recoverable": self.recoverable,
            "retained_bytes": self.retained_bytes,
            "fragment_bytes": self.fragment_bytes,
        }


def conditional_residual(fragment: str, retained: str, *, compressor: str = "zlib") -> float:
    """CCR(F | R) for a single compressor.

    Returns 1.0 for an empty retained set (nothing can be recovered) and
    0.0 for an empty fragment (nothing is lost).
    """
    if not fragment.strip():
        return 0.0
    frag_b = fragment.encode("utf-8")
    if not retained.strip():
        return 1.0
    ret_b = retained.encode("utf-8")

    c_ret = _adjusted(compressor, ret_b)
    c_joint = _adjusted(compressor, ret_b + _SEP + frag_b)
    c_frag = _adjusted(compressor, frag_b)

    marginal = max(0, c_joint - c_ret)
    return min(1.0, marginal / c_frag)


def certify_recoverable(
    fragment: str,
    retained: str,
    *,
    threshold: float = 0.55,
    compressors: tuple[str, ...] = ("zlib", "bz2", "lzma"),
) -> CompressionCertificate:
    """Recoverability certificate, fail-closed across the compressor ensemble.

    The reported residual is the **maximum** over compressors: a fragment is
    certified recoverable only when every codec agrees it is cheap given the
    retained set.
    """
    scores = tuple(
        (name, conditional_residual(fragment, retained, compressor=name))
        for name in compressors
    )
    deciding, residual = max(scores, key=lambda kv: kv[1])
    return CompressionCertificate(
        residual=residual,
        per_compressor=scores,
        deciding_compressor=deciding,
        threshold=threshold,
        recoverable=residual <= threshold,
        retained_bytes=len(retained.encode("utf-8")),
        fragment_bytes=len(fragment.encode("utf-8")),
    )


@dataclass(frozen=True)
class ContainmentCertificate:
    """Heuristic comparison against each codec's self-residual.

    ``contained`` records a threshold decision, not logical containment or
    semantic entailment. ``sigma`` is a multiplier, not a statistical interval.
    """

    contained: bool
    gap: float
    noise_floor: float
    margin: float
    per_compressor: tuple[tuple[str, float, float], ...]
    deciding_compressor: str
    residual: float

    def to_dict(self) -> dict:
        return {
            "contained": self.contained,
            "gap": round(self.gap, 4),
            "noise_floor": round(self.noise_floor, 4),
            "margin": round(self.margin, 4),
            "per_compressor": {
                k: {"gap": round(g, 4), "delta": round(d, 4)}
                for k, g, d in self.per_compressor
            },
            "deciding_compressor": self.deciding_compressor,
            "residual": round(self.residual, 4),
        }


def _self_residual(text: str, compressor: str) -> float:
    """CCR(F|F) — noise floor for a single compressor."""
    if not text.strip():
        return 0.0
    b = text.encode("utf-8")
    c_f = _adjusted(compressor, b)
    c_ff = _adjusted(compressor, b + _SEP + b)
    marginal = max(0, c_ff - c_f)
    return min(1.0, marginal / c_f)


def certify_containment(
    fragment: str,
    retained: str,
    *,
    sigma: float = 2.0,
    compressors: tuple[str, ...] = ("zlib", "bz2", "lzma"),
) -> ContainmentCertificate:
    """Compare redundancy gap to scaled self-residual across codecs.

    The smallest margin decides. No calibrated probability or mathematical
    containment guarantee follows from this heuristic.
    """
    per = []
    for name in compressors:
        ccr = conditional_residual(fragment, retained, compressor=name)
        delta = _self_residual(fragment, name)
        gap = 1.0 - ccr
        per.append((name, gap, delta))

    deciding_name, deciding_gap, deciding_delta = min(per, key=lambda t: t[1] - sigma * t[2])
    margin = deciding_gap - sigma * deciding_delta
    return ContainmentCertificate(
        contained=margin > 0.0,
        gap=deciding_gap,
        noise_floor=deciding_delta,
        margin=margin,
        per_compressor=tuple(per),
        deciding_compressor=deciding_name,
        residual=1.0 - deciding_gap,
    )


def asymmetry(text_a: str, text_b: str, *, compressor: str = "zlib") -> tuple[float, float]:
    """Return (CCR(A|B), CCR(B|A)).

    A large gap records directional byte redundancy, which can suggest
    subsumption but cannot establish it.  Symmetric
    similarity measures collapse this to a single number and lose it.
    """
    return (
        conditional_residual(text_a, text_b, compressor=compressor),
        conditional_residual(text_b, text_a, compressor=compressor),
    )


# Temporal compression diagnostics; codec shifts do not prove semantic safety.

@dataclass(frozen=True)
class ContextDriftReceipt:
    """Measurements for a proposed append, requiring semantic re-verification.

    ``stable`` is only True for an empty omission or unchanged byte-contained
    evidence. Otherwise it is None (unknown). ``collapse_detected`` is a legacy
    name for codec similarity, not proof of information loss or future failure.
    """

    stable: bool | None
    pre_update_residual: float
    post_update_residual: float
    residual_shift: float
    shift_threshold: float
    collapse_detected: bool
    per_compressor: tuple[tuple[str, float, float], ...]
    requires_reverification: bool
    reason: str
    input_sha256: tuple[str, str, str]

    def to_dict(self) -> dict:
        from dataclasses import asdict
        result = asdict(self)
        result['per_compressor'] = {
            name: {'pre': pre, 'post': post, 'shift': post - pre}
            for name, pre, post in self.per_compressor
        }
        result['scope'] = 'compression diagnostic, not semantic omission safety'
        return result


def measure_context_drift(
    fragment: str, retained: str, update: str, *, shift_threshold: float = 0.15,
    compressors: tuple[str, ...] = ('zlib', 'bz2', 'lzma'),
) -> ContextDriftReceipt:
    """Measure CCR before and after appending update; abstain on safety.

    Codec windowing and framing can change marginal compressed size. Neither
    the sign nor magnitude proves that an update requires omitted evidence.
    Every nonempty update invalidates this diagnostic's unchanged-input case.
    """
    import hashlib
    import math
    if not math.isfinite(shift_threshold) or shift_threshold < 0:
        raise ValueError('shift_threshold must be finite and nonnegative')
    if not compressors or any(name not in COMPRESSORS for name in compressors):
        raise ValueError('at least one supported compressor is required')
    updated = retained + _SEP.decode() + update if update else retained
    per = tuple((name, conditional_residual(fragment, retained, compressor=name),
                 conditional_residual(fragment, updated, compressor=name))
                for name in compressors)
    _, pre, post = max(per, key=lambda row: row[2] - row[1])
    similarity = bool(update) and all(
        conditional_residual(retained, updated, compressor=name) < 0.1 and
        conditional_residual(updated, retained, compressor=name) < 0.1
        for name in compressors
    )
    trivial = not fragment or (not update and fragment in retained)
    reason = ('empty_omission' if not fragment else
              'unchanged_byte_containment' if trivial else 'requires_reverification')
    return ContextDriftReceipt(
        stable=True if trivial else None, pre_update_residual=pre,
        post_update_residual=post, residual_shift=post - pre,
        shift_threshold=shift_threshold, collapse_detected=similarity,
        per_compressor=per, requires_reverification=not trivial, reason=reason,
        input_sha256=tuple(hashlib.sha256(text.encode()).hexdigest()
                           for text in (fragment, retained, update)),
    )
