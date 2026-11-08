"""compress() on JSON must return JSON, and mark what it left out.

The README's own example, ``compress(api_response, budget=2000)`` on 500
``{"id", "status"}`` items, returned text that began ``"ok"}, {"id": 20`` and
ended mid-object: the schema pass declined, ``universal_compress`` handed the
JSON to the prose sentence extractor, and items 0-19 and 240-459 vanished
without a marker.
"""

from __future__ import annotations

import json
import re

from entroly import compress
from entroly.sdk import count_tokens
from entroly.universal_compress import compress_json_structure, universal_compress


def _readme_payload(error_at: int | None = None) -> str:
    items = [{"id": i, "status": "ok"} for i in range(500)]
    if error_at is not None:
        items[error_at] = {"id": error_at, "status": "error", "message": "payment declined"}
    return json.dumps({"items": items})


def _accounted_items(text: str) -> int:
    """Items kept plus items an omission marker says it replaced."""
    kept = len(re.findall(r'"id":\s*\d+', text))
    omitted = sum(int(n) for n in re.findall(r"\[\.\.\. (\d+) items? omitted \.\.\.\]", text))
    return kept + omitted


def test_readme_example_returns_valid_json_within_budget() -> None:
    out = compress(_readme_payload(), budget=2000)

    parsed = json.loads(out)
    assert isinstance(parsed, dict) and "items" in parsed
    assert count_tokens(out) <= 2000
    assert parsed["items"][0] == {"id": 0, "status": "ok"}
    assert parsed["items"][-1] == {"id": 499, "status": "ok"}


def test_every_item_is_kept_or_counted_by_a_marker() -> None:
    for budget in (2000, 400, 100):
        out = compress(_readme_payload(error_at=250), budget=budget)
        json.loads(out)
        assert _accounted_items(out) == 500, (budget, out[:300])


def test_an_error_item_in_the_middle_survives() -> None:
    out = compress(_readme_payload(error_at=250), budget=100)
    assert "payment declined" in out


def test_universal_json_never_falls_back_to_the_prose_extractor() -> None:
    text = _readme_payload()
    out, ctype, _savings = universal_compress(text, 0.3, "json")
    assert ctype == "json"
    json.loads(out)


def test_structure_compressor_returns_input_when_it_already_fits() -> None:
    data = {"a": [1, 2, 3]}
    assert json.loads(compress_json_structure(data, max_chars=1000)) == data
