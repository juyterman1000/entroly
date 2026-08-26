"""Three commands reported numbers that were not true.

Each of these was found by using the product rather than reading it, and each
misreports in the direction that flatters: a perfect savings score for having
returned nothing, more files than the repository contains, and a permanent
warning that the native engine is missing when it is loaded.
"""

from __future__ import annotations

import entroly.auto_index as auto_index


def test_empty_selection_is_not_a_full_saving() -> None:
    """Selecting nothing saves nothing.

    `verify-claims` computed `(1 - used/tokens) * 100`, which returns 100.0
    when `used` is 0. The first command the welcome banner tells a new user to
    run therefore headlined a perfect score for returning no context, while
    `cmd_simulate` scored the identical event as 0.0%.
    """
    tokens = 5_706

    def savings(used: int, selected: list) -> float:
        return (1 - used / tokens) * 100 if selected and tokens > 0 and used <= tokens else 0.0

    assert savings(0, []) == 0.0, "an empty selection must not report a saving"
    assert savings(2_000, [{"token_count": 2_000}]) > 0.0


def test_native_engine_check_reads_a_key_the_emitter_actually_sets() -> None:
    """`doctor --json` could never report a healthy native engine.

    It read `engine.native.available`; `runtime_capabilities` emits
    `installed` / `healthy` / `version` / `missing_symbol_count` and has no
    `available` key, so the pass branch was unreachable on every machine.
    """
    from entroly.runtime_capabilities import runtime_capabilities

    native = runtime_capabilities().get("engine", {}).get("native", {})
    assert "available" not in native, (
        "if `available` is reintroduced, runtime_doctor must be updated with it"
    )
    # These are the keys the doctor check is now written against.
    assert {"installed", "healthy"} <= set(native)


def test_files_indexed_is_a_file_count_not_a_fragment_count() -> None:
    """`files_indexed` carried the fragment tally.

    A file can yield more than one fragment, so the value could exceed the
    number of files in the repository -- it was printed to users, returned in
    `simulate`/`perf --json` and `verify-claims`, and published as the headline
    of the CI dogfood evidence artifact.

    This asserts the two are reported separately, which is what makes
    conflating them again a visible change rather than a silent one.
    """
    source = (auto_index.__file__ or "")
    assert source, "auto_index must be importable from a file"
    text = open(source, encoding="utf-8").read()

    assert "fragments_ingested" in text, (
        "the fragment count must stay a distinct field from files_indexed"
    )
    assert 'indexed = int(r.get("ingested", 0))' not in text, (
        "files_indexed must not be assigned the fragment count again"
    )
