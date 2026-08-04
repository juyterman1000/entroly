"""Context construction for the agentic-task arms.

Separated from the runner on purpose: these functions are pure, so the part
that decides *what the model sees* can be tested without a model, a network,
or a git worktree. The runner only sequences them.

The three arms differ in exactly one respect -- the context handed to the
model. Everything else (model, decoding parameters, seed, prompt template,
test oracle, token accounting) is held identical, because the question is
whether compression changes the outcome, not whether some other knob does.

    RAW              every candidate file, verbatim, in a fixed order
    COMPRESS         Entroly's task-conditioned selection under a token budget
    CLOSED_LOOP      COMPRESS, plus recovery of omitted spans on a retry

Entroly never alters generation parameters; see the output-only contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

__all__ = [
    "Arm",
    "Fragment",
    "BuiltContext",
    "build_raw",
    "build_compressed",
    "build_closed_loop",
    "build_for_arm",
    "estimate_tokens",
]


class Arm(str, Enum):
    """The preregistered arms. Values match the artifact's `mode` field."""

    RAW = "raw"
    COMPRESS = "entroly_compress_only"
    CLOSED_LOOP = "entroly"


@dataclass(frozen=True)
class Fragment:
    """One candidate piece of context, with its provenance."""

    source: str
    content: str

    @property
    def digest(self) -> str:
        """Content hash, so an omitted fragment can be identified later."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_engine_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "token_count": estimate_tokens(self.content),
        }


@dataclass
class BuiltContext:
    """What an arm decided to show the model, and what it held back.

    `omitted` is not discarded: the closed-loop arm reads it on retry, and the
    artifact records it so a failure can be attributed to a specific dropped
    span rather than to compression in the abstract.
    """

    arm: Arm
    text: str
    included: list[Fragment] = field(default_factory=list)
    omitted: list[Fragment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "estimated_context_tokens": self.estimated_tokens,
            "included_sources": [f.source for f in self.included],
            "omitted_sources": [f.source for f in self.omitted],
            "omitted_digests": [f.digest for f in self.omitted],
            "notes": list(self.notes),
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimate, used only for budgeting.

    Measured token counts always come from the provider's own accounting
    (`prompt_eval_count`), never from this function. Keeping the two separate
    stops a budgeting heuristic from silently becoming a reported result.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _render(fragments: Sequence[Fragment]) -> str:
    """Deterministic rendering, so arms differ by content and not by layout."""
    blocks = [
        f"--- {fragment.source} ---\n{fragment.content}" for fragment in fragments
    ]
    return "\n\n".join(blocks)


def build_raw(fragments: Sequence[Fragment]) -> BuiltContext:
    """Every fragment, verbatim. The control arm."""
    ordered = list(fragments)
    return BuiltContext(
        arm=Arm.RAW,
        text=_render(ordered),
        included=ordered,
        omitted=[],
        notes=["full context, no selection"],
    )


def build_compressed(
    fragments: Sequence[Fragment],
    *,
    query: str,
    budget: int,
    optimize_fn: Callable[..., dict[str, Any]],
) -> BuiltContext:
    """Entroly's task-conditioned selection under a token budget.

    `optimize_fn` is injected rather than imported so the arm can be tested
    against a stub. In the runner it is `entroly.optimize`, i.e. the same path
    the SDK, MCP `optimize_context`, and proxy injection all use -- not a
    benchmark-only reimplementation.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")

    payload = [fragment.to_engine_dict() for fragment in fragments]
    result = optimize_fn(payload, budget=budget, query=query)

    selected_sources = _selected_sources(result)
    by_source = {fragment.source: fragment for fragment in fragments}

    # Preserve the engine's ordering where it gave one; fall back to input
    # order so the arm stays deterministic even if the engine does not rank.
    included = [by_source[s] for s in selected_sources if s in by_source]
    if not included:
        # Fail loudly rather than silently degrading into the RAW arm: an
        # empty selection is a real finding about the engine, not a reason to
        # quietly hand the model everything and call it "compressed".
        raise RuntimeError(
            "optimize() selected no fragments; refusing to fall back to full "
            "context, which would report a compression result that never "
            "compressed anything"
        )

    included_sources = {fragment.source for fragment in included}
    omitted = [f for f in fragments if f.source not in included_sources]

    return BuiltContext(
        arm=Arm.COMPRESS,
        text=_render(included),
        included=included,
        omitted=omitted,
        notes=[f"budget={budget}", f"selected {len(included)}/{len(fragments)}"],
    )


def build_closed_loop(
    base: BuiltContext,
    *,
    recovered: Sequence[Fragment],
) -> BuiltContext:
    """COMPRESS plus spans recovered after a failed first attempt.

    This is the arm that tests whether recovery earns its cost: it only runs
    when the first attempt failed, and it is charged for both attempts.
    """
    recovered_list = [f for f in recovered if f.source not in {
        frag.source for frag in base.included
    }]
    included = list(base.included) + recovered_list
    still_omitted = [
        f for f in base.omitted
        if f.source not in {frag.source for frag in recovered_list}
    ]
    return BuiltContext(
        arm=Arm.CLOSED_LOOP,
        text=_render(included),
        included=included,
        omitted=still_omitted,
        notes=list(base.notes) + [f"recovered {len(recovered_list)} omitted span(s)"],
    )


def build_for_arm(
    arm: Arm,
    fragments: Sequence[Fragment],
    *,
    query: str,
    budget: int,
    optimize_fn: Callable[..., dict[str, Any]] | None = None,
) -> BuiltContext:
    """Dispatch to the arm's builder.

    CLOSED_LOOP's first attempt is identical to COMPRESS by construction --
    recovery is what happens *after* a failure, so the two arms must start
    from the same context or the comparison is not paired.
    """
    if arm is Arm.RAW:
        return build_raw(fragments)
    if optimize_fn is None:
        raise ValueError(f"arm {arm.value} requires optimize_fn")
    built = build_compressed(
        fragments, query=query, budget=budget, optimize_fn=optimize_fn
    )
    if arm is Arm.COMPRESS:
        return built
    if arm is Arm.CLOSED_LOOP:
        return BuiltContext(
            arm=Arm.CLOSED_LOOP,
            text=built.text,
            included=built.included,
            omitted=built.omitted,
            notes=list(built.notes) + ["first attempt (pre-recovery)"],
        )
    raise ValueError(f"unknown arm: {arm!r}")


def _selected_sources(result: dict[str, Any]) -> list[str]:
    """Read the selected sources out of an optimize() result.

    optimize() has carried its selection under more than one key across
    versions. Rather than guess, this checks the known shapes and raises if
    none match -- a silent empty list here would look like "compression
    dropped everything" and corrupt the measurement.
    """
    if not isinstance(result, dict):
        raise TypeError(f"optimize() returned {type(result).__name__}, expected dict")

    # An empty selection and an unrecognised payload are different failures:
    # the first is a finding about the engine, the second is a shape change
    # this reader has not been taught. Collapsing them would let a rename look
    # like "compression dropped everything".
    recognised_key = False
    for key in ("selected", "fragments", "selected_fragments", "context"):
        value = result.get(key)
        if not isinstance(value, list):
            continue
        recognised_key = True
        sources: list[str] = []
        for item in value:
            if isinstance(item, dict):
                source = item.get("source") or item.get("path") or item.get("id")
                if isinstance(source, str):
                    sources.append(source)
            elif isinstance(item, str):
                sources.append(item)
        if sources:
            return sources

    if recognised_key:
        return []  # genuinely empty selection; the caller decides what that means

    raise KeyError(
        "could not find selected fragments in optimize() result; "
        f"top-level keys were {sorted(result)}"
    )
