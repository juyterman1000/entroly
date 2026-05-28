"""SRP DIFF resolution — change-driven read mode.

The DIFF resolution lets SRP render only the modified hunks of a block
when the caller passes ``previous_source``. This is the change-driven
read-mode counterpart to FULL/MEDIUM/LOW/SKIP: ideal when the agent
needs to learn *what changed* without re-paying for unchanged code.

Each test below pins one piece of the contract so regressions show up
as concrete assertion failures, not silent behavior drift.
"""

from __future__ import annotations

from entroly.semantic_resolution import Resolution, resolve


SRC_V1 = """\
def login(user, pwd):
    if not validate(user):
        return None
    return token(user)


def logout(uid):
    cache.remove(uid)


def healthcheck():
    return "ok"
"""

SRC_V2 = """\
def login(user, pwd):
    if not validate(user, pwd):
        return None
    return token(user, pwd)


def logout(uid):
    cache.remove(uid)


def healthcheck():
    return "ok"
"""


def test_diff_resolution_exists_in_enum():
    assert Resolution.DIFF == "diff"
    # Cost must sit between LOW and MEDIUM — diff is cheaper than
    # signature+docstring but richer than a bare stub.
    assert Resolution.COST["low"] < Resolution.COST["diff"] < Resolution.COST["medium"]


def test_previous_source_omitted_means_diff_never_selected():
    """Without previous_source, DIFF is never assigned."""
    r = resolve(source=SRC_V2, query="login fix", budget=400, file_path="auth.py")
    assert "diff" not in r.resolution_counts or r.resolution_counts.get("diff", 0) == 0


def test_modified_block_renders_as_diff_under_tight_budget():
    """When the same file is passed as previous_source but tightened on
    budget, the modified `login` block should appear as DIFF — the
    cheapest level that still conveys the change."""
    r = resolve(
        source=SRC_V2,
        previous_source=SRC_V1,
        query="anything",
        budget=80,  # tight enough that FULL won't fit everywhere
        file_path="auth.py",
    )
    # Across all rendered blocks, at least one DIFF emission is expected.
    assert r.resolution_counts.get("diff", 0) >= 1, (
        f"expected at least one DIFF block, got {r.resolution_counts}"
    )
    # The diff output for `login` must surface the change.
    assert "+" in r.output and "validate(user, pwd)" in r.output


def test_unchanged_blocks_never_render_as_diff():
    """`logout` and `healthcheck` are byte-identical across versions, so
    no diff exists for them. They must NOT be assigned DIFF resolution
    (DIFF requires a non-empty delta)."""
    r = resolve(
        source=SRC_V2,
        previous_source=SRC_V1,
        query="logout",
        budget=300,
        file_path="auth.py",
    )
    for rb in r.blocks:
        if rb.block.name in ("logout", "healthcheck"):
            assert rb.resolution != Resolution.DIFF, (
                f"unchanged block {rb.block.name} got DIFF resolution"
            )


def test_diff_output_includes_signature_anchor():
    """A pure unified diff has no class/function-name context. SRP must
    prepend the signature line so the LLM has a structural anchor."""
    r = resolve(
        source=SRC_V2,
        previous_source=SRC_V1,
        query="anything",
        budget=80,
        file_path="auth.py",
    )
    # Find any DIFF block in the result
    diff_rb = next(
        (rb for rb in r.blocks if rb.resolution == Resolution.DIFF and rb.output),
        None,
    )
    assert diff_rb is not None, "no DIFF block was emitted"
    # Output must start with the function signature (the anchor),
    # not with the `---` / `+++` unified-diff prelude.
    first_line = diff_rb.output.splitlines()[0]
    assert "def " in first_line or "class " in first_line, (
        f"DIFF output should lead with the signature, got: {first_line!r}"
    )


def test_diff_total_tokens_respects_budget():
    """The budget contract holds whether DIFF is enabled or not —
    final total_tokens must not exceed the requested budget."""
    r = resolve(
        source=SRC_V2,
        previous_source=SRC_V1,
        query="login",
        budget=50,
        file_path="auth.py",
    )
    assert r.total_tokens <= 50, f"DIFF mode broke budget: {r.total_tokens} > 50"


def test_new_block_in_v2_falls_back_to_standard_ladder():
    """A block that exists only in the new version has no previous
    counterpart, so DIFF is impossible — it must render at FULL/MEDIUM/
    LOW/SKIP per the standard ladder, never DIFF."""
    src_v1 = "def foo():\n    return 1\n"
    src_v2 = (
        "def foo():\n    return 1\n\n\n"
        "def bar():\n    return 2\n"
    )
    r = resolve(
        source=src_v2,
        previous_source=src_v1,
        query="bar",
        budget=200,
        file_path="x.py",
    )
    bar_rb = next((rb for rb in r.blocks if rb.block.name == "bar"), None)
    assert bar_rb is not None
    assert bar_rb.resolution != Resolution.DIFF, (
        "newly-added block should not render as DIFF (no previous_source for it)"
    )
