"""Conditional compression residual: a model-free recoverability certificate.

An omission is recoverable when the retained set already contains the
omitted fragment's information.  Formally, fragment F is recoverable from
retained set R when the conditional Kolmogorov complexity K(F | R) is near
zero.  K is uncomputable, but Cilibrasi & Vitanyi (2005) showed that real
compressors approximate it:

    K(F | R)  ~=  C(R + F) - C(R)

Normalising by F's own compressed size gives the **conditional compression
residual**:

    CCR(F | R) = [C(R + F) - C(R)] / C(F)

    CCR -> 0   F costs almost nothing given R; R already carries it.
    CCR -> 1   F is fully novel given R.

Why this and not embedding cosine similarity:

1. **Asymmetry.**  Omission safety is directional.  A short fragment may be
   fully recoverable from a longer one while the reverse is false.  Cosine
   similarity is symmetric and cannot represent that distinction at all;
   this is a representational limit, not a tuning limit.
2. **No model download.**  ``zlib``, ``bz2`` and ``lzma`` are standard
   library, preserving local-first operation.
3. **Determinism.**  Byte-identical output forever, at a fixed level.
   Embedding scores drift across model versions, which breaks receipt
   replay.
4. **Explainability.**  "Appending F to R cost 12 of 87 bytes" is an
   inspectable receipt line; a cosine of 0.83 is not.

Ensemble rule: compressors have different inductive biases (LZ77 literal
repeats, BWT reordering, LZMA long-range structure).  A safety certificate
must take the **maximum** residual across compressors, never the mean --
the most conservative compressor decides, so a single permissive codec can
never certify an unsafe omission.

This module answers *recoverability* only.  It cannot answer *constraint
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
    """Replayable recoverability certificate.

    Every field needed to recompute the verdict is recorded, so an auditor
    can reproduce it from the fragment bytes alone.
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


def asymmetry(text_a: str, text_b: str, *, compressor: str = "zlib") -> tuple[float, float]:
    """Return (CCR(A|B), CCR(B|A)).

    A large gap between the two is the signature of subsumption: one text
    contains the other's information but not the reverse.  Symmetric
    similarity measures collapse this to a single number and lose it.
    """
    return (
        conditional_residual(text_a, text_b, compressor=compressor),
        conditional_residual(text_b, text_a, compressor=compressor),
    )
