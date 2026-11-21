"""A receipt's token budget must be certified in real tokens.

The native ingest sized chunks and reported ``token_count`` with
``estimate_tokens``, which counts matches of ``[A-Za-z0-9][A-Za-z0-9_']*`` and
therefore ignores every bracket, operator, quote, newline and indent -- roughly
half the token stream in source code. Selection spent the caller's budget in
those private units, so a receipt reported ``selected_tokens == token_budget``
for text that was nearly twice the budget, and the constraint it certified
could not be violated. Measured before the fix on this repository: 1,200
claimed against 2,311 real o200k tokens, 1.93x.

The undercount is content-dependent -- prose, Python, Rust and JSON all differ
-- so no constant could correct it. The pipeline now ingests, restates the
accounting with the same tokenizer the Python path already used, and only then
selects, which makes the budget the engine enforces the same one the receipt
reports.

These tests measure against the real tokenizer, never against a stored
constant, so they stay true if chunking or ranking changes.
"""

from __future__ import annotations

import pytest

from entroly.sdk import create_context_receipt
from entroly.tokens import count_tokens, tokenizer_is_exact

pytestmark = pytest.mark.skipif(
    not tokenizer_is_exact(),
    reason="exact token accounting needs the real encoder installed",
)

#: Punctuation-dense source, where counting word runs undercounts the worst.
_CORPUS = {
    "service.py": (
        "import json\n"
        "from dataclasses import dataclass, field\n"
        "\n"
        "@dataclass(slots=True)\n"
        "class Settings:\n"
        "    retries: int = 3\n"
        "    backoff: float = 0.5\n"
        "    tags: list[str] = field(default_factory=list)\n"
        "\n"
        "    def as_dict(self) -> dict[str, object]:\n"
        "        return {'retries': self.retries, 'backoff': self.backoff}\n"
        "\n"
        "def load(path):\n"
        "    with open(path, encoding='utf-8') as handle:\n"
        "        return Settings(**json.load(handle))\n"
    ) * 4,
    "payload.json": (
        '{"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], '
        '"meta": {"count": 2, "next": null}}\n'
    ) * 12,
    "notes.md": (
        "The capture path retries on transient failures and records the "
        "idempotency key so a replay cannot double charge.\n"
    ) * 8,
}


def _selected_text(receipt) -> str:
    return "".join(c.get("text", "") for c in (receipt.get("selected_context") or []))


def _claimed_tokens(receipt) -> int:
    return sum(
        int(c.get("token_count", 0) or 0)
        for c in (receipt.get("selected_context") or [])
    )


@pytest.mark.parametrize("budget", [400, 1200])
def test_the_selection_fits_the_budget_in_real_tokens(budget) -> None:
    """The defect: the budget was saturated in units nothing else used."""
    receipt = create_context_receipt(
        _CORPUS, query="retry backoff idempotency key", budget=budget,
        chunk_tokens=150, overlap_tokens=16,
    )
    exact = count_tokens(_selected_text(receipt))
    assert exact <= budget, (
        f"selected {exact} real tokens against a {budget}-token budget; "
        "the certificate is measured in units the constraint cannot violate"
    )


@pytest.mark.parametrize("prefer_rust", [True, False])
def test_the_reported_count_is_the_real_count(prefer_rust) -> None:
    """What the receipt claims must be what a provider would bill."""
    receipt = create_context_receipt(
        _CORPUS, query="retry backoff idempotency key", budget=1200,
        chunk_tokens=150, overlap_tokens=16, prefer_rust=prefer_rust,
    )
    claimed = _claimed_tokens(receipt)
    exact = count_tokens(_selected_text(receipt))
    assert claimed > 0 and exact > 0
    # Joining chunk texts can shift a token or two at the seams; the defect
    # was a ~2x systematic undercount, not a rounding difference.
    assert abs(claimed - exact) <= max(2, exact // 50), (
        f"receipt claims {claimed} tokens for text that is {exact} "
        f"({exact / claimed:.2f}x)"
    )


def test_both_backends_agree_on_the_accounting() -> None:
    """The two paths must not report the same corpus in different units."""
    kwargs = dict(query="retry backoff idempotency key", budget=1200,
                  chunk_tokens=150, overlap_tokens=16)
    native = create_context_receipt(_CORPUS, prefer_rust=True, **kwargs)
    python = create_context_receipt(_CORPUS, prefer_rust=False, **kwargs)
    for receipt in (native, python):
        claimed = _claimed_tokens(receipt)
        exact = count_tokens(_selected_text(receipt))
        assert abs(claimed - exact) <= max(2, exact // 50), (
            f"backend disagrees with the tokenizer: {claimed} vs {exact}"
        )


def test_restating_the_counts_keeps_recovery_byte_exact(tmp_path) -> None:
    """Only the accounting is restated; offsets and digests stay the engine's.

    Guards the trust invariant directly: if restating had disturbed chunk
    boundaries or digests, recovery would stop reproducing the original bytes.
    """
    receipt = create_context_receipt(
        _CORPUS, query="retry backoff idempotency key", budget=400,
        chunk_tokens=150, overlap_tokens=16,
        recoverable=True, store_dir=str(tmp_path),
    )
    from entroly.sdk import recover_receipt_omission

    recovered = recover_receipt_omission(receipt, store_dir=str(tmp_path))
    assert recovered, "nothing was omitted, so this proves nothing"
    for chunk in recovered:
        assert chunk.get("verified") is True
        source = _CORPUS.get(str(chunk.get("source_path", "")), "")
        assert chunk.get("text", "") in source, "recovered text is not the original"


def test_token_offsets_share_the_unit_of_token_count() -> None:
    """token_start/token_end must not silently keep the estimator's unit."""
    receipt = create_context_receipt(
        _CORPUS, query="retry backoff idempotency key", budget=1200,
        chunk_tokens=150, overlap_tokens=16,
    )
    chunks = receipt.get("selected_context") or []
    assert chunks
    for chunk in chunks:
        start, end = chunk.get("token_start"), chunk.get("token_end")
        if start is None or end is None:
            continue
        assert end - start == int(chunk.get("token_count", 0) or 0), (
            "token offsets and token_count disagree, so one of them is "
            "still counting word runs"
        )
