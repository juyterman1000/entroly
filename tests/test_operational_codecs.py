from __future__ import annotations

from pathlib import Path

from entroly.codec import RecoveryStore
from entroly.codecs_operational import DiffCodec, HtmlCodec, SearchResultCodec


def _store(tmp_path: Path) -> RecoveryStore:
    return RecoveryStore(tmp_path / "recovery.json", scope_id="operational-codecs-test")


def test_diff_codec_keeps_all_changes_and_recovers_exact_source(tmp_path: Path) -> None:
    context = "".join(f" context {index}\n" for index in range(80))
    source = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,80 +1,80 @@\n" + context + "-old = 1\n+new = 2\n"
    store = _store(tmp_path)
    reps = DiffCodec(store).representations(source, source_id="patch")
    compressed = min(reps, key=lambda row: row.token_cost)
    assert "-old = 1" in compressed.text
    assert "+new = 2" in compressed.text
    assert compressed.recovery is not None
    assert store.recover(compressed.recovery) == source


def test_search_codec_groups_and_recovers_hits(tmp_path: Path) -> None:
    source = "\n".join(f"src/a.py:{index}:result value {index}" for index in range(1, 40)) + "\n"
    store = _store(tmp_path)
    reps = SearchResultCodec(store).representations(source, source_id="rg", max_hits_per_file=3)
    compressed = min(reps, key=lambda row: row.token_cost)
    assert "additional hit" in compressed.text
    assert compressed.recovery is not None
    assert store.recover(compressed.recovery) == source


def test_search_codec_preserves_first_seen_file_order_and_unparsed_failures(tmp_path: Path) -> None:
    source = (
        "z.py:1:first\n"
        + "\n".join(f"z.py:{index}:noise" for index in range(2, 15))
        + "\na.py:1:second\nmetadata one\nmetadata two\nmetadata three\nFATAL unparsed failure\n"
    )
    reps = SearchResultCodec(_store(tmp_path)).representations(
        source, source_id="rg", max_hits_per_file=1
    )
    compressed = min(reps, key=lambda row: row.token_cost)
    assert compressed.text.index("z.py:1:first") < compressed.text.index("a.py:1:second")
    assert "FATAL unparsed failure" in compressed.text


def test_html_codec_ignores_scripts_retains_query_and_recovers(tmp_path: Path) -> None:
    noise = "".join(f"<p>catalog row {index}</p>" for index in range(200))
    source = f"<html><head><title>Billing</title><script>ignore-secret()</script></head><body><h1>Invoice settings</h1><button>Save invoice</button>{noise}</body></html>"
    store = _store(tmp_path)
    reps = HtmlCodec(store).representations(source, source_id="page", query="billing invoice", budget=80)
    compressed = min(reps, key=lambda row: row.token_cost)
    assert "Billing" in compressed.text
    assert "Invoice" in compressed.text
    assert "ignore-secret" not in compressed.text
    assert compressed.recovery is not None
    assert store.recover(compressed.recovery) == source
