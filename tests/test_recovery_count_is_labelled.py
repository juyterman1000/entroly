"""A count printed next to a payload must say what it counts.

`RecoveryReference.item_count` means something different in every codec, by
design and pinned by `test_codec_contract`: `json` counts the records it elided
(40 records -> 39), `log` counts every line in the original, the shell codec
counts non-empty lines removed. All of them were printed as a bare "N item(s)",
appended to a note stating the payload was the *complete original*.

Measured by running the documented flow on a 60-record JSON file: compress,
then `entroly recover <digest>`. The bytes came back byte-identical -- the
reversibility contract held -- and the message read

    complete original JSON for records.json (...) (59 item(s))

The file has 60 records. The number was true (59 were elided) and the sentence
it appeared in was not. `item_label` makes each codec say what it counted;
the number itself is unchanged, because it is a contract other tests pin.
"""

from __future__ import annotations

import json

from entroly.codec import RecoveryReference, RecoveryStore
from entroly.codecs_builtin import JsonCodec, LogCodec
from entroly.codecs_table import TableCodec


def _records(n: int = 60) -> str:
    return json.dumps(
        [{"id": i, "user": f"u{i}", "status": ["ok", "fail"][i % 2]} for i in range(n)],
        indent=2,
    )


def _recovery_for(codec, text: str, source_id: str):
    reps = [r for r in codec.representations(text, source_id=source_id) if r.recovery]
    assert reps, "codec produced no recoverable representation"
    return reps[0].recovery


class TestTheLabelDescribesTheNumber:
    def test_json_says_the_count_is_of_elided_records(self):
        recovery = _recovery_for(JsonCodec(RecoveryStore()), _records(), "records.json")

        assert recovery.item_label != "item(s)", (
            "the default label next to 'complete original JSON' is what described "
            "a 60-record payload as 59 items"
        )
        assert "elid" in recovery.item_label
        assert "record" in recovery.item_label

    def test_the_json_count_itself_is_unchanged(self):
        """The number is a contract; only its label was wrong."""
        recovery = _recovery_for(JsonCodec(RecoveryStore()), _records(60), "r.json")
        assert recovery.item_count == 59

    def test_log_says_its_count_is_of_lines_not_of_elided_items(self):
        text = "\n".join(
            f"2026-08-02T10:00:{i % 60:02d}Z ERROR request failed (retry {i})"
            for i in range(200)
        )
        recovery = _recovery_for(LogCodec(RecoveryStore()), text, "worker.log")
        assert "line" in recovery.item_label

    def test_a_table_count_says_it_excludes_the_header(self):
        text = "id,name\n" + "\n".join(f"{i},row{i}" for i in range(40))
        recovery = _recovery_for(TableCodec(RecoveryStore()), text, "t.csv")
        assert "row" in recovery.item_label and "header" in recovery.item_label


class TestTheLabelSurvivesTheStore:
    def test_it_round_trips_through_put_and_rehydration(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
        store = RecoveryStore()
        ref = store.put("original bytes", item_count=3, item_label="widget(s) dropped")

        assert ref.item_label == "widget(s) dropped"
        assert ref.to_dict()["item_label"] == "widget(s) dropped"

    def test_a_reference_written_before_this_field_still_reads(self):
        """Old stores carry no label; they must not crash or print 'None'."""
        ref = RecoveryReference(digest="sha256:" + "0" * 64, byte_length=1, item_count=5)
        assert ref.item_label == "item(s)"
        assert ref.to_dict()["item_label"] == "item(s)"


class TestRecoveryIsStillExact:
    def test_labelling_did_not_disturb_the_bytes(self):
        """The reversibility contract is the thing that must not move."""
        text = _records(60)
        store = RecoveryStore()
        recovery = _recovery_for(JsonCodec(store), text, "records.json")

        recovered = store.recover(recovery)
        assert recovered == text
        assert recovery.verify(recovered)
