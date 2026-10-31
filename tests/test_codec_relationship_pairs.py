"""Compressed forms must keep related facts related.

A codec can pass every existing check -- each value still present somewhere,
the original exactly recoverable -- and still hand the model a view where it
can no longer tell which outcome belongs to which test, which error code
belongs to which file:line, or which value belongs to which row. That view is
worse than omission: it invites a confident wrong answer.

Each fixture carries *pairs* of facts that sit together in the source. For
every representation the default registry offers, the representation is
decoded into the units a reader would see as associated -- a line, a record
rebuilt from a template or a columnar table -- and a pair whose two halves both
survive must survive in the same unit, with no other partner in its place. A
form that drops both halves is a different matter (the elision is visible and
recoverable); a form that keeps both but separates them fails.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterable

import pytest

from entroly.codec import RecoveryStore, Representation
from entroly.codecs_builtin import default_registry

Pair = tuple[str, str]
Fixture = tuple[str, list[Pair], frozenset[str]]


# ── fixtures: (text, pairs, the vocabulary a wrong partner would come from) ──


def pytest_output() -> Fixture:
    lines, pairs = [], []
    outcomes = frozenset({"PASSED", "FAILED", "SKIPPED"})
    for i in range(300):
        outcome = "FAILED" if i in (37, 141, 260) else ("SKIPPED" if i % 97 == 5 else "PASSED")
        name = f"tests/test_mod{i % 9}.py::test_case_{i:03d}"
        lines.append(f"{name} {outcome}")
        if outcome != "PASSED":
            pairs.append((name, outcome))
    lines.append("=== 3 failed, 294 passed, 3 skipped in 4.2s ===")
    return "\n".join(lines), pairs, outcomes


def lint_output() -> Fixture:
    lines, pairs = [], []
    codes = ["E501", "F401", "W291", "E302"]
    for i in range(250):
        location = f"entroly/mod_{i % 13}.py:{10 + i}:{1 + i % 7}"
        code = "F821" if i in (77, 190) else codes[i % 4]
        lines.append(f"{location}: {code} message number {i}")
        if code == "F821":
            pairs.append((location, code))
    return "\n".join(lines), pairs, frozenset({*codes, "F821"})


def csv_table() -> Fixture:
    rng = random.Random(7)
    rows, pairs, values = ["id,status,value"], [], set()
    for i in range(200):
        value = str(rng.randint(10_000, 99_999))
        values.add(value)
        status = "error" if i in (12, 150) else "ok"
        rows.append(f"row-{i:04d},{status},{value}")
        if status == "error":
            pairs.append((f"row-{i:04d}", value))
    return "\n".join(rows), pairs, frozenset(values)


def json_records() -> Fixture:
    rng = random.Random(11)
    records, pairs, values = [], [], set()
    for i in range(120):
        value = rng.randint(10_000, 99_999)
        values.add(str(value))
        failed = i in (3, 88)
        records.append({"id": f"rec-{i:04d}", "status": "error" if failed else "ok", "value": value})
        if failed:
            pairs.append((f"rec-{i:04d}", str(value)))
    return json.dumps(records, indent=2), pairs, frozenset(values)


def worker_log() -> Fixture:
    """Templated lines where both halves of every pair vary per line."""
    rng = random.Random(3)
    lines, pairs, values = [], [], set()
    for i in range(240):
        job = f"job-{i:04d}"
        took = f"{rng.randint(100, 9_999)}ms"
        values.add(took)
        lines.append(f"2026-10-31T10:{i // 60:02d}:{i % 60:02d}Z INFO worker finished {job} took={took}")
        if i % 40 == 7:
            pairs.append((job, took))
    return "\n".join(lines), pairs, frozenset(values)


def tagged_requests() -> Fixture:
    """Slot values that contain the template column separator ``|``."""
    rng = random.Random(5)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
    users = ["ann", "bob", "cyd", "dee", "eve"]
    lines, pairs, tags = [], [], set()
    for i in range(120):
        tag = rng.choice(words) + ("|" + rng.choice(words) if i % 2 else "")
        user = f"{rng.choice(users)}{i:03d}"
        tags.add(f"tags={tag}")
        lines.append(f"INFO handled user={user} tags={tag} region=eu")
        if i % 15 == 1:
            pairs.append((f"user={user}", f"tags={tag}"))
    return "\n".join(lines), pairs, frozenset(tags)


FIXTURES: dict[str, Callable[[], Fixture]] = {
    "tagged-requests": tagged_requests,
    "pytest": pytest_output,
    "lint": lint_output,
    "csv": csv_table,
    "json": json_records,
    "worker-log": worker_log,
}


# ── decoding a representation into the units a reader sees as associated ──


_TEMPLATE = re.compile(r"^\[x(\d+)\] (.*)$")
_COLUMN = re.compile(r"^    \{(\d+)\}: (.*)$")


def _template_units(text: str) -> list[str]:
    """Rebuild every line a templated log form stands for."""
    templates = [m for m in map(_TEMPLATE.match, text.splitlines()) if m]
    columns = [m.group(2).split("|") for m in map(_COLUMN.match, text.splitlines()) if m]
    if not templates:
        return []
    units, cursor = [], 0
    for match in templates:
        count, template = int(match.group(1)), match.group(2)
        parts = template.split("{}")
        slots = columns[cursor:cursor + len(parts) - 1]
        cursor += len(parts) - 1
        assert all(len(column) == count for column in slots), (
            f"template {template!r} has {count} lines but its value columns do not",
        )
        for row in range(count):
            rebuilt = parts[0]
            for column, part in zip(slots, parts[1:]):
                rebuilt += column[row] + part
            units.append(rebuilt)
    assert cursor == len(columns), "value columns left over: they belong to no template"
    return units


def _json_units(text: str) -> list[str]:
    try:
        payload = json.loads(text)
    except ValueError:
        return []
    units: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("_format") == "columnar":
                names = [name for name in node.get("_verbatim_columns", []) if name in node]
                length = int(node.get("_records", 0))
                for name in names:
                    assert len(node[name]) == length, f"column {name!r} is misaligned"
                units.extend(
                    json.dumps({name: node[name][i] for name in names}) for i in range(length)
                )
                return
            units.append(json.dumps(node))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return units


def units_of(rep: Representation) -> list[str]:
    return [*rep.text.splitlines(), *_template_units(rep.text), *_json_units(rep.text)]


def _has(unit: str, token: str) -> bool:
    return re.search(rf"(?<![\w|-]){re.escape(token)}(?![\w|-])", unit) is not None


def broken_pairs(units: Iterable[str], pairs: list[Pair], vocabulary: frozenset[str]) -> list[str]:
    units = list(units)
    problems = []
    for left, right in pairs:
        if not (any(_has(u, left) for u in units) and any(_has(u, right) for u in units)):
            continue  # one half was elided: visible, recoverable, not a mismatch
        holding = [unit for unit in units if _has(unit, left)]
        if not any(_has(unit, right) for unit in holding):
            problems.append(f"{left!r} kept, but no longer next to {right!r}")
        for unit in holding:
            wrong = sorted(token for token in vocabulary - {right} if _has(unit, token))
            if wrong:
                problems.append(f"{left!r} appears beside the wrong partner {wrong}")
    return problems


def _representations(name: str) -> tuple[Fixture, list[Representation]]:
    fixture = FIXTURES[name]()
    registry = default_registry(RecoveryStore())
    return fixture, list(registry.representations(fixture[0], source_id=name))


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_representation_keeps_pairs_together(name: str) -> None:
    (_, pairs, vocabulary), reps = _representations(name)
    assert len(reps) >= 2, "fixture must exercise at least one compressed form"
    failures = {
        rep.representation_id: problems
        for rep in reps
        if (problems := broken_pairs(units_of(rep), pairs, vocabulary))
    }
    assert not failures, failures


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_a_compressed_form_is_offered_and_decodes(name: str) -> None:
    """The check above is only meaningful if something was compressed."""
    (text, _, _), reps = _representations(name)
    compressed = [rep for rep in reps if rep.text != text]
    assert compressed, f"no compressed representation for {name}"
    for rep in compressed:
        units_of(rep)  # decoders assert alignment as they go


# ── the checker itself must catch a separated pair ───────────────────────


def test_checker_catches_a_pair_split_across_units() -> None:
    text = "tests/a.py::test_one PASSED\ntests/a.py::test_two FAILED"
    shuffled = "tests/a.py::test_one\ntests/a.py::test_two\nFAILED PASSED"
    pairs = [("tests/a.py::test_two", "FAILED")]
    vocabulary = frozenset({"PASSED", "FAILED"})
    assert not broken_pairs(text.splitlines(), pairs, vocabulary)
    assert broken_pairs(shuffled.splitlines(), pairs, vocabulary)


def test_checker_catches_a_misaligned_template_column() -> None:
    good = "[x2] job-{} took={}\n    {0}: 0001|0002\n    {1}: 5ms|9ms"
    swapped = "[x2] job-{} took={}\n    {0}: 0001|0002\n    {1}: 9ms|5ms"
    pairs = [("job-0001", "5ms")]
    vocabulary = frozenset({"5ms", "9ms"})
    assert not broken_pairs(_template_units(good), pairs, vocabulary)
    assert broken_pairs(_template_units(swapped), pairs, vocabulary)


def test_checker_catches_misaligned_columnar_json() -> None:
    columnar = json.dumps({"_format": "columnar", "_records": 2, "_verbatim_columns": ["id", "v"],
                           "id": ["r1", "r2"], "v": [7, 8]})
    swapped = columnar.replace("[7, 8]", "[8, 7]")
    pairs = [("r1", "7")]
    vocabulary = frozenset({"7", "8"})
    assert not broken_pairs(_json_units(columnar), pairs, vocabulary)
    assert broken_pairs(_json_units(swapped), pairs, vocabulary)
