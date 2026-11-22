"""Default native deduplication must retain this behavior-changing code edit."""

from types import SimpleNamespace

import pytest


def test_default_native_dedup_preserves_refund_decision_change():
    core = pytest.importorskip("entroly_core")
    original = (
        "def allow_refund(order, account):\n"
        "    if order.is_cancelled:\n"
        "        return False\n"
        "    if order.is_fraudulent:\n"
        "        return False\n"
        "    return True\n"
    )
    revised = original.replace("    return True\n", "    return False\n")
    order = SimpleNamespace(is_cancelled=False, is_fraudulent=False)
    decisions = []
    for content in (original, revised):
        scope = {}
        exec(content, scope)
        decisions.append(scope["allow_refund"](order, None))
    assert decisions == [True, False]
    assert (core.py_simhash(original) ^ core.py_simhash(revised)).bit_count() == 6

    # This pair is the reason the hard drop no longer consults distance.
    #
    # d = 6 sits inside even the widest threshold tried, and the two functions
    # return different answers for the same order, so a distance rule merged
    # them and discarded a real behaviour change. It could not do otherwise:
    # a one-token edit moves the same number of trigrams whether or not the
    # token carries meaning, so no threshold separates this from a rename.
    #
    # Dropping now requires content equality, so the revision survives at
    # every threshold -- asserted at 8 because that was the value that lost it.
    wider = core.PyDedupIndex(8)
    assert wider.insert("original", original) is None
    assert wider.insert("revised", revised) is None

    # The distance is still available, just not as grounds for deletion.
    # Querying by text, so "revised" matches itself at 0; "original" is the
    # nearest *other* fragment and must be reported with its true distance.
    probe = core.PyDedupIndex(8)
    assert probe.insert("original", original) is None
    nearest = probe.nearest_within_threshold(revised)
    assert nearest == ("original", 6), f"expected ('original', 6), got {nearest}"

    engine = core.EntrolyEngine()
    assert engine.ingest(original, "refund.py", 0, False)["status"] == "ingested"
    assert engine.ingest(revised, "refund.py", 0, False)["status"] == "ingested"
