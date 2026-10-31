from __future__ import annotations

import re
import symtable
import types

import pytest

from entroly.repository_intelligence import rename_binding_review as review_module
from entroly.repository_intelligence.rename_binding_review import review_python_rename_bindings


def _edit(before: bytes, selected=None):
    matches = list(re.finditer(rb"\bcharge\b", before))
    changes = [{"start_byte": item.start(), "end_byte": item.end(),
                "old_identifier": "charge", "new_identifier": "authorize"}
               for index, item in enumerate(matches) if selected is None or index in selected]
    after = before
    for item in reversed(changes):
        after = after[:item["start_byte"]] + b"authorize" + after[item["end_byte"]:]
    return after, changes


def test_equal_scope_owners_do_not_equate_two_different_local_parameters():
    before = b"def charge(amount):\n return amount + 2\ndef other(charge, authorize):\n return charge(3)\n"
    after, changes = _edit(before, selected={0, 2})
    review = review_python_rename_bindings("payments.py", before, after, changes)
    assert review["conflicts"]
    assert review["semantic_status"] == "not-checked"


def test_unmapped_import_alias_migration_is_unknown_not_a_false_conflict():
    before = b"import math as charge\ndef process(amount):\n return charge.sqrt(amount)\n"
    after, changes = _edit(before)
    review = review_python_rename_bindings("payments.py", before, after, changes)
    assert review["status"] == "unknown"
    assert not review["conflicts"]


def test_nonlocal_assignment_does_not_own_the_outer_cell():
    source = "def make():\n balance = 0\n def relay():\n  nonlocal balance\n  balance = 3\n  def read():\n   return balance\n  return read\n return relay\n"
    namespace = {}
    exec(compile(source, "<authored-closure>", "exec"), namespace)
    outer = namespace["make"].__code__
    middle = next(item for item in outer.co_consts if isinstance(item, types.CodeType))
    assert "balance" in outer.co_cellvars
    assert "balance" in middle.co_freevars and "balance" not in middle.co_cellvars
    tables = review_module._tables(symtable.symtable(source, "closure.py", "exec"))
    make = next(path for path, table in tables.items() if table.get_name() == "make")
    read = next(path for path, table in tables.items() if table.get_name() == "read")
    assert review_module._owner(tables, read, "balance") == ("scope", make)


def test_duplicate_edits_are_rejected():
    before = b"def charge(amount):\n return amount\n"
    after, changes = _edit(before)
    with pytest.raises(ValueError, match="duplicate"):
        review_python_rename_bindings("payments.py", before, after, changes * 2)


def test_postimage_must_match_exact_edits():
    before = b"def charge(amount):\n return amount\n"
    after, changes = _edit(before)
    with pytest.raises(ValueError, match="exact edits"):
        review_python_rename_bindings("payments.py", before, after + b"\n", changes)


def test_review_never_executes_module_statements():
    before = b"raise AssertionError('analysis executed source')\ndef charge(amount):\n return amount\n"
    after, changes = _edit(before)
    review = review_python_rename_bindings("payments.py", before, after, changes)
    assert review["semantic_status"] == "not-checked"


def test_retained_conflicts_are_bounded_without_losing_conflict_status(monkeypatch):
    before = b"def charge(amount):\n return amount\ndef process(authorize):\n return charge(1) + charge(2) + charge(3)\n"
    after, changes = _edit(before)
    monkeypatch.setattr(review_module, "MAX_RETAINED_CONFLICTS", 1)
    review = review_python_rename_bindings("payments.py", before, after, changes)
    assert review["conflict_count"] == 3
    assert len(review["conflicts"]) == 1
    assert review["omitted_conflict_count"] == 2
    assert review["status"] == "conflict"


def test_source_limit_yields_unknown_not_clear(monkeypatch):
    before = b"def charge(amount):\n return amount\n"
    after, changes = _edit(before)
    monkeypatch.setattr(review_module, "MAX_REVIEW_SOURCE_BYTES", 8)
    review = review_python_rename_bindings("payments.py", before, after, changes)
    assert review["status"] == "unknown"
    assert "source-size-limit" in review["unknown"]
    assert review["semantic_status"] == "not-checked"


@pytest.mark.parametrize("path", ("payments.py", "payments.pyi", "payments.pyw"))
def test_supported_python_extensions_carry_conflict_evidence(path):
    before = b"def charge(amount):\n return amount\nauthorize = 5\n"
    after, changes = _edit(before)
    assert review_python_rename_bindings(path, before, after, changes)["conflict_count"] > 0


def test_reflection_can_change_behavior_without_a_lexical_conflict():
    before = b"def tag(fn):\n fn.label = fn.__name__\n return fn\n@tag\ndef charge(amount):\n return amount\ndef process():\n return charge.label\n"
    after, changes = _edit(before)
    review = review_python_rename_bindings("payments.py", before, after, changes)
    assert not review["conflicts"]
    assert review["semantic_status"] == "not-checked"
    left, right = {}, {}
    exec(compile(before, "<authored-before>", "exec"), left)
    exec(compile(after, "<authored-after>", "exec"), right)
    assert left["process"]() != right["process"]()
