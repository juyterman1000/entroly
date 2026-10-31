"""An interrogative is a task, not a factual premise for the answer.

These deterministic regressions exercise evidence eligibility, not semantic
code verification, a sampled accuracy rate, or the private reader smoke.
"""
from __future__ import annotations

import pytest

from entroly.esg import ESGAnalyzer, _split_atoms
from entroly.sdk import eicv_verify


@pytest.mark.parametrize(
    ("question", "claim"),
    [
        ("Does Vault encrypt backups?", "Vault encrypts backups."),
        ("Is Ledger append only?", "Ledger is append only."),
        ("Can Gateway reject unsigned requests", "Gateway can reject unsigned requests."),
        ("What does normalize(True) return?", "normalize(True) returns True."),
        ("Does Vault encrypt backups? Is Ledger append only?", "Vault encrypts backups."),
    ],
)
def test_a_question_alone_does_not_supply_a_factual_evidence_atom(question, claim):
    result = ESGAnalyzer().score(question, claim)

    assert result.n_ev_atoms == 0
    assert result.mean_max_support == 0.0
    assert result.unsupported_fraction == 1.0
    assert all(not verdict.best_ev_atom for verdict in result.atom_verdicts)


@pytest.mark.parametrize("separator", [" ", "\n", "\r\n"])
@pytest.mark.parametrize("question_first", [True, False])
def test_a_question_cannot_override_a_contradicting_source(separator, question_first):
    question = "What does normalize(True) return?"
    source = "normalize returns None for boolean input."
    evidence = separator.join((question, source) if question_first else (source, question))
    claim = "normalize(True) returns True."
    expected = ESGAnalyzer().score(source, claim)
    actual = ESGAnalyzer().score(evidence, claim)

    assert actual.as_dict() == expected.as_dict()


@pytest.mark.parametrize("question", ["Is Vault enabled", "Does Vault encrypt backups"])
@pytest.mark.parametrize("question_first", [True, False])
def test_an_unpunctuated_question_line_does_not_consume_adjacent_source(question, question_first):
    source = "Vault encrypts backups."
    evidence = "\n".join((question, source) if question_first else (source, question))
    claim = "Vault encrypts backups."

    assert ESGAnalyzer().score(evidence, claim).as_dict() == ESGAnalyzer().score(source, claim).as_dict()


@pytest.mark.parametrize(
    "source",
    [
        "When the lock times out, the append is unserialized.",
        "Where the path is absolute, resolution is skipped.",
        "What the receipt omits is listed under omitted_context.",
        "How this differs is in the token budget.",
        "Which engine serves is decided by native_status.",
        "Why this matters is that the record is unverifiable.",
    ],
)
def test_declarative_wh_clauses_are_still_evidence(source):
    result = ESGAnalyzer().score(source, source)

    assert result.n_ev_atoms >= 1
    assert result.mean_max_support == 1.0
    assert result.tension == 0.0


@pytest.mark.parametrize("source", ["None", "42", "OK"])
def test_the_short_assertion_fallback_does_not_resurrect_a_question(source):
    expected = ESGAnalyzer().score(source, source)
    actual = ESGAnalyzer().score("What is the result? " + source, source)

    assert actual.as_dict() == expected.as_dict()


def test_inline_fact_before_a_question_is_not_dropped_with_its_line():
    source = "Vault encrypts backups."
    mixed = source + " Does Ledger preserve history?"

    assert ESGAnalyzer().score(mixed, source).as_dict() == ESGAnalyzer().score(source, source).as_dict()


def test_the_public_sdk_cannot_certify_an_answer_from_its_question():
    cert = eicv_verify(
        "normalize(True) returns True.",
        evidence="What does normalize(True) return?",
        profile="qa",
    )

    assert cert["decision"] != "supported"
    assert cert["n_ev_atoms"] == 0
    assert cert["unsupported_fraction"] == 1.0


def test_empty_reference_stays_unsupported():
    result = ESGAnalyzer().score("", "Vault encrypts backups.")

    assert result.n_ev_atoms == 0
    assert result.unsupported_fraction == 1.0


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r"])
def test_filtering_does_not_synthesize_a_premise_across_a_question(separator):
    text = separator.join(("Vault", "Does Ledger preserve history?", "encrypts"))

    assert _split_atoms(text, exclude_questions=True) == ["Vault", "encrypts"]


@pytest.mark.parametrize(
    "text",
    [
        "None",
        "42",
        "Vault encrypts backups. Ledger preserves history.",
        "The parser binds input\nto the unique parameter.",
        "def normalize(value):\n    if value is None:\n        return 0\n    return value\n",
        "ValueError is caught before the fallback is returned.",
        "The string contains '?' as a literal.",
        "The path is /api/search?query=cache.",
    ],
)
def test_reference_decomposition_is_unchanged_without_questions(text):
    assert _split_atoms(text, exclude_questions=True) == _split_atoms(text)


def test_a_question_cannot_supply_communication_auto_action_support():
    from entroly.communication import (
        CommunicationActionProposal,
        action_evidence_verification,
        eicv_supports_automatic_action,
        event_from_adapter,
    )

    event = event_from_adapter({
        "direction": "inbound", "channel": "test", "account_id": "local",
        "conversation_id": "evidence-roles", "conversation_kind": "direct",
        "sender_id": "person", "message_id": "question", "timestamp": 1,
        "content": "Does Vault encrypt backups?", "source": "test",
    })
    payload = "Vault encrypts backups."
    proposal = CommunicationActionProposal.build(
        action_type="reply", channel="test", account_id="local",
        conversation_id="evidence-roles", conversation_kind="direct",
        source_event_ids=[event.event_id], payload=payload, category="other",
    )

    verification = action_evidence_verification(proposal, [event], payload=payload)

    assert verification["status"] == "verified"
    assert verification["decision"] != "supported"
    assert not eicv_supports_automatic_action(verification)
