"""`compress()` must keep scaling with its input, and must not splice.

`_compress_code` handed whole source files to `py_compress_block`, which
compresses one *conversation block* -- note the "assistant" role it is given.
On a file it returns a head fragment, a literal "[...]", and a tail fragment,
sized independently of both the input and the requested budget.

Measured over the 56 files in `entroly/` larger than 2 KB at ratio 0.3, the
native result was a median of 3% of the requested size and never above 25%;
`cli.py` compressed 302,338 characters to 84. Output stopped growing with input
altogether -- 67 KB and 21 MB both returned about 100 characters -- so the loss
was unbounded in the size of the input.

It was also misleading rather than merely lossy. Joining a head to a tail across
an elided middle reads as contiguous code: on distinct handlers it returned
`handler_0000`'s signature attached to `handler_2599`'s return statement, so a
reader is told handler_0000 returns route 2599. A plausible false fact is worse
than dropped content, and this is the SDK's public entry point.
"""

from __future__ import annotations

from entroly.sdk import compress


def _distinct_handlers(count: int) -> str:
    """Genuinely distinct functions.

    Repeating one file would not prove anything: a deduplicating compressor is
    entitled to collapse it. Every function here is unique.
    """
    return "\n".join(
        f"def handler_{i:05d}(request, session):\n"
        f"    total_{i} = request.get({i}) * {i + 7}\n"
        f'    return {{"route": {i}}}\n'
        for i in range(count)
    )


def test_output_grows_with_input() -> None:
    small = compress(_distinct_handlers(200))
    large = compress(_distinct_handlers(2000))

    # A ten-fold larger corpus must not compress to the same size. The bug
    # returned ~734 characters for both.
    assert len(large) > len(small) * 3, (
        f"output saturated: {len(small)} chars for 200 handlers vs "
        f"{len(large)} for 2000"
    )


def test_large_input_is_not_reduced_to_a_fixed_excerpt() -> None:
    source = _distinct_handlers(2000)
    out = compress(source)

    # Not a byte-exact ratio assertion: the compactor is allowed judgement.
    # This only rejects collapse to a constant-size excerpt, which is what
    # made loss unbounded in input size.
    assert len(out) > len(source) * 0.05, (
        f"kept {len(out) / len(source):.4%} of a {len(source):,}-char corpus"
    )


def test_does_not_splice_one_function_onto_another() -> None:
    """The output must not attach one function's body to another's signature."""
    out = compress(_distinct_handlers(2000))

    # Every retained `return {"route": N}` must belong to a handler_N that is
    # still the nearest preceding signature. The bug produced
    # `def handler_00000(...)` followed by `return {"route": 2599}`.
    current: str | None = None
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("def handler_"):
            current = stripped[len("def handler_") : len("def handler_") + 5]
        elif stripped.startswith('return {"route":') and current is not None:
            route = stripped.split(":")[1].strip().rstrip("}").strip()
            assert route == str(int(current)), (
                f"spliced: handler_{current} shown returning route {route}"
            )


def test_smaller_ratio_keeps_less() -> None:
    source = _distinct_handlers(500)
    generous = compress(source, target_ratio=0.3)
    frugal = compress(source, target_ratio=0.1)

    # The bug had only two reachable outputs across every requested ratio.
    assert len(frugal) < len(generous), (
        f"target_ratio ignored: 0.1 -> {len(frugal)}, 0.3 -> {len(generous)}"
    )


def test_input_already_within_budget_is_returned_unchanged() -> None:
    """The existing non-annihilation contract must still hold."""
    tiny = "def f():\n    return 1\n"
    assert compress(tiny, budget=10_000) == tiny
    assert compress("") == ""
