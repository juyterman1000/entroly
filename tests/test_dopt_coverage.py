import pytest

pytest.importorskip("numpy")

from entroly.dopt_selector import select  # noqa: E402


def fragment(name, source, content, tokens=20):
    return {"fragment_id": name, "source": source, "content": content, "token_count": tokens}


def test_geometry_repairs_redundancy_with_a_rare_caller():
    definition = fragment(
        "definition", "src/reconcile.py", "def reconcile(record):\n    return canonicalize(record)\n"
    )
    repeat = fragment(
        "repeat", "src/reconcile.py", "if record:\n    canonicalize(record)\n    reconcile(record)\n"
    )
    caller = fragment(
        "caller", "src/recovery.py", "def recover_expired_lease(lease):\r\n    return reconcile(lease.checkpoint)\r\n"
    )
    filename_only = fragment(
        "unrelated", "docs/tool/reconcile.md", "# Orchard\nApples, pears, and plums.\n"
    )
    candidates = [filename_only, definition, repeat, caller]

    baseline = select(candidates, 40, "reconcile")
    repaired = select(candidates, 40, "reconcile", coverage_repair=True)

    assert baseline == [definition, repeat]
    assert repaired == [definition, caller]
    assert repaired[0] is definition
    assert repaired[1] is caller
    assert b"\r\n" in repaired[1]["content"].encode()
    assert sum(raw["token_count"] for raw in repaired) == 40
    assert select(candidates[:-1], 40, "reconcile", coverage_repair=True) == baseline


@pytest.mark.parametrize("tokens", [20, 21])
def test_explicit_symbol_repair_fits_fixed_budget_and_prefers_code(tokens):
    prefix = fragment(
        "prefix", "src/lease.py", "def expire(lease):\n    return lease.deadline < clock.monotonic()\n"
    )
    tail = fragment(
        "tail", "src/lease.py", "def release(resource):\n    audit.emit(resource.disposition)\n"
    )
    mentions = fragment(
        "mentions", "src/lease.py", '# recover_expired_lease(lease)\nexample = """\nrecover_expired_lease(lease)\n"""\n'
    )
    near_match = fragment(
        "near-match", "src/lease.py", "def recover_expired_leases(lease):\n    return lease\n"
    )
    caller = fragment(
        "caller", "src/lease.py", "def poll(scheduler):\n    return scheduler.recover_expired_lease(scheduler.oldest)\n",
        tokens,
    )
    query = "recover_expired_lease recovery"
    candidates = [prefix, tail, mentions, near_match, caller]

    assert select(candidates, 40, query) == [prefix, tail]
    repaired = select(candidates, 40, query, coverage_repair=True)

    assert repaired == [prefix, caller if tokens == 20 else tail]
    assert sum(raw["token_count"] for raw in repaired) == 40


def test_symbol_repair_reserves_space_for_a_callable_in_a_partial_chunk():
    prefix = fragment("prefix", "src/lease.py", "def expire(lease):\n    return lease.deadline\n")
    first_tail = fragment("first-tail", "src/lease.py", "    clock.tick()\n", 10)
    last_tail = fragment("last-tail", "src/lease.py", "    audit.emit(record)\n", 10)
    caller = fragment(
        "caller", "src/lease.py", "        audit.emit(lease)\n    return recover_expired_lease(lease)\n", 13
    )
    candidates = [prefix, first_tail, last_tail, caller]

    assert select(candidates, 40, "recover_expired_lease") == [prefix, first_tail, last_tail]
    repaired = select(candidates, 40, "recover_expired_lease", coverage_repair=True)

    assert repaired == [prefix, caller]
    assert repaired[1] is caller
    assert sum(raw["token_count"] for raw in repaired) == 33


@pytest.mark.parametrize(("prefix_text", "tail_text"), [
    ("# recover expired lease\n", "# audit\n"),
    ("# recover expired lease audit_events\n", "# AUDIT_events\n"),
])
def test_symbol_repair_preserves_unique_tail_terms_and_exact_anchors(prefix_text, tail_text):
    prefix = fragment(
        "prefix", "src/lease.py", prefix_text
    )
    tail = fragment("tail", "src/lease.py", tail_text)
    caller = fragment("caller", "src/lease.py", "def poll(lease):\n    return recover_expired_lease(lease)\n")
    query = "recover_expired_lease AUDIT_events"
    candidates = [prefix, tail, caller]

    assert select(candidates, 40, query) == [prefix, tail]
    assert select(candidates, 40, query, coverage_repair=True) == [prefix, tail]


def test_empty_query_keeps_existing_selection():
    candidates = [fragment("first", "a.py", "def read_checkpoint(lease):\n    return lease.checkpoint\n")]

    assert select(candidates, 20, coverage_repair=True) == select(candidates, 20)
