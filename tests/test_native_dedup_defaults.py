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

    # The experimental wider threshold loses a meaningful revision despite
    # strong lexical similarity. Keep this negative control beside the default.
    wider = core.PyDedupIndex(8)
    assert wider.insert("original", original) is None
    assert wider.insert("revised", revised) == "original"

    engine = core.EntrolyEngine()
    assert engine.ingest(original, "refund.py", 0, False)["status"] == "ingested"
    assert engine.ingest(revised, "refund.py", 0, False)["status"] == "ingested"
