"""compress_messages must not hide a failing test, and must show what it cut.

A pytest log in a tool message (hundreds of PASSED lines, then the failure,
its assertion and the run summary) came back as PASSED lines only, with no
sign anything was removed, at budget=30000 and budget=8000. An agent reading
that conversation would conclude the tests pass. The selected spans were
joined without gap markers, and tool messages were cut to 0.3 of the share
the budget allowed, so 30,000 tokens of budget produced about 9,000.
"""

from __future__ import annotations

import copy
import re

from entroly import compress_messages
from entroly.tokens import count_tokens

PASSED = "".join(
    f"tests/test_basic.py::test_case_{i} PASSED [{i % 100:3d}%]\n" for i in range(700)
)
FAILURE = (
    "tests/test_options.py::test_flag_default FAILED\n"
    "=================================== FAILURES ===================================\n"
    "    def test_flag_default():\n"
    ">       assert result.output == 'True\\n'\n"
    "E       AssertionError: assert 'False\\n' == 'True\\n'\n"
    "tests/test_options.py:412: AssertionError\n"
    "==== 1 failed, 700 passed in 3.21s ====\n"
)
LOG = PASSED + FAILURE
CODE = "".join(
    f"def helper_{i}(value):\n    \"\"\"Return value plus {i}.\"\"\"\n    return value + {i}\n\n"
    for i in range(1500)
)
JS = "".join(
    f"res.method{i} = function method{i}(body) {{\n  return this.send(body + {i});\n}};\n\n"
    for i in range(400)
)
SYSTEM = "You are the release agent. Always run the tests before claiming success."
MARKER = re.compile(r"\[\.\.\. [^\]]*(?:omitted|condensed)[^\]]* \.\.\.\]|\[truncated\]|aged-pruned")


def _conversation(log_first: bool = True):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "CI is red. Find the failing test and fix it."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_run", "type": "function", "function": {"name": "run_tests", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_run", "content": LOG},
        {"role": "assistant", "content": "Reading the sources.", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"core.py\"}"}},
            {"id": "call_b", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"response.js\"}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": CODE},
        {"role": "tool", "tool_call_id": "call_b", "content": JS},
        {"role": "assistant", "content": "The default for flag options is computed in get_default."},
        {"role": "user", "content": "OK, patch Option.get_default so flag defaults are honoured and rerun the tests."},
    ]


def _assert_failure_kept(text: str) -> None:
    for evidence in ("test_flag_default FAILED", "AssertionError: assert 'False", "1 failed, 700 passed"):
        assert evidence in text, f"{evidence!r} was dropped from the test output: ...{text[-300:]!r}"


def test_failing_test_survives_compression_at_any_budget():
    for budget in (30000, 8000, 3000):
        out = compress_messages(copy.deepcopy(_conversation()), budget=budget)
        _assert_failure_kept(out[3]["content"])


def test_every_shortened_message_says_so():
    original = _conversation()
    for budget in (30000, 8000):
        out = compress_messages(copy.deepcopy(original), budget=budget)
        for before, after in zip(original, out):
            if isinstance(before.get("content"), str) and after.get("content") != before.get("content"):
                assert MARKER.search(after["content"]), (
                    f"budget={budget}: {before['role']} message shortened from {len(before['content'])} "
                    f"to {len(after['content'])} chars with no omission marker"
                )


def test_budget_is_spent_before_evidence_is_cut():
    conversation = _conversation()
    source = sum(count_tokens(m["content"]) for m in conversation if isinstance(m.get("content"), str))
    budget = 30000
    assert source > 1.5 * budget
    out = compress_messages(copy.deepcopy(conversation), budget=budget)
    total = sum(count_tokens(m["content"]) for m in out if isinstance(m.get("content"), str))
    assert total <= budget
    assert total >= 0.6 * budget, f"kept {total:,} tokens of a {budget:,} budget"


def test_aged_tool_output_keeps_its_failure():
    conversation = _conversation()
    later = []
    for i in range(5):
        later += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_x{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"call_x{i}", "content": f"file {i}\n" + "x = 1\n" * 200},
        ]
    conversation = conversation[:-1] + later + conversation[-1:]
    out = compress_messages(copy.deepcopy(conversation), budget=20000)
    _assert_failure_kept(out[3]["content"])
