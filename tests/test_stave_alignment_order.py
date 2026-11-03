"""Source-order regressions using hand-declared facts, without model calls.

Mass fixtures describe distinct people at one instant, in the same unit, with
exactly one mass each. Crossed masses contradict those declared facts. Unknown
people are unsupported controls; no closed-world truth label is inferred.
"""

from itertools import permutations

import pytest

from entroly import witness
from entroly.verifiers.stave import Triplet, _align_one, stave_risk, stave_verify
from entroly.witness import Certificate, ProofStep, WitnessAnalyzer
from entroly.witness_risk_model import RiskModel


MASS_FACTS = (
    "Alice weighs 60 kilograms.",
    "Bruno weighs 80 kilograms.",
    "Clara weighs 90 kilograms.",
)
PEOPLE_VALUES = (("Alice", "60"), ("Bruno", "80"), ("Clara", "90"))
TWO_ORDERS = tuple(permutations(MASS_FACTS[:2]))
THREE_ORDERS = tuple(permutations(MASS_FACTS))


@pytest.mark.parametrize("facts", TWO_ORDERS)
@pytest.mark.parametrize("answer", MASS_FACTS[:2])
def test_exact_supported_fact_has_zero_risk_in_both_orders(facts, answer):
    result = stave_verify(answer, " ".join(facts))
    assert len(result.knowledge_triplets) == 2
    assert len(result.answer_triplets) == len(result.alignments) == 1
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == answer
    assert alignment.pred_sim == alignment.subject_fid == alignment.object_fid == 1.0
    assert alignment.rsf == 1.0
    assert result.hallucination_risk == stave_risk(answer, " ".join(facts)) == 0.0
    assert result.verdict == "pass"
    assert not result.wrong_slot_detected


@pytest.mark.parametrize("facts", TWO_ORDERS)
@pytest.mark.parametrize(
    "answer,owner", (("Alice weighs 80 kilograms.", MASS_FACTS[0]),
                     ("Bruno weighs 60 kilograms.", MASS_FACTS[1])),
)
def test_crossed_value_and_wrong_subject_do_not_borrow_slots(facts, answer, owner):
    result = stave_verify(answer, " ".join(facts))
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == owner
    assert alignment.subject_fid == 1.0
    assert alignment.object_fid == alignment.rsf == 0.0
    assert alignment.wrong_slot
    assert result.hallucination_risk == stave_risk(answer, " ".join(facts)) == 1.0
    assert result.n_wrong_slot == 1
    assert result.verdict == "flag"


@pytest.mark.parametrize("facts", THREE_ORDERS)
@pytest.mark.parametrize("person,value", PEOPLE_VALUES)
def test_all_three_entity_permutations_preserve_supported_fact(facts, person, value):
    answer = f"{person} weighs {value} kilograms."
    result = stave_verify(answer, " ".join(facts))
    assert {(t.subject, t.predicate, t.object_) for t in result.knowledge_triplets} == {
        (p, "have", v) for p, v in PEOPLE_VALUES
    }
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == answer
    assert alignment.subject_fid == alignment.object_fid == alignment.rsf == 1.0
    assert result.hallucination_risk == 0.0
    assert not result.wrong_slot_detected


@pytest.mark.parametrize("facts", THREE_ORDERS)
@pytest.mark.parametrize(
    "person,wrong_value,owner_value",
    tuple((person, wrong, correct) for person, correct in PEOPLE_VALUES
          for _, wrong in PEOPLE_VALUES if wrong != correct),
)
def test_all_three_entity_permutations_reject_crossed_values(
    facts, person, wrong_value, owner_value,
):
    result = stave_verify(f"{person} weighs {wrong_value} kilograms.", " ".join(facts))
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.subject == person
    assert alignment.knowledge_triplet.object_ == owner_value
    assert alignment.subject_fid == 1.0
    assert alignment.object_fid == alignment.rsf == 0.0
    assert alignment.wrong_slot
    assert result.hallucination_risk == 1.0


@pytest.mark.parametrize("facts", THREE_ORDERS)
@pytest.mark.parametrize(
    "answer,wrong_slot",
    (("Alice weighs 77 kilograms.", True),
     ("Dorian weighs 60 kilograms.", True),
     ("Dorian weighs 77 kilograms.", False)),
)
def test_wrong_value_and_unknown_subject_controls(facts, answer, wrong_slot):
    result = stave_verify(answer, " ".join(facts))
    assert len(result.answer_triplets) == len(result.alignments) == 1
    assert result.n_matched == 1
    assert result.hallucination_risk == 1.0
    assert result.wrong_slot_detected is wrong_slot
    assert result.verdict == "flag"
    if answer.startswith("Alice"):
        assert result.alignments[0].knowledge_triplet.subject == "Alice"
    elif "60" in answer:
        assert result.alignments[0].knowledge_triplet.object_ == "60"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Bruno is Ruby.", "Alice belongs Amber."))),
)
def test_exact_predicate_precedes_better_slots_on_partial_predicate(facts):
    # be/belong has prefix similarity 0.7; be/be has exact similarity 1.0.
    result = stave_verify("Alice is Amber.", " ".join(facts))
    assert {(t.subject, t.predicate, t.object_) for t in result.knowledge_triplets} == {
        ("Bruno", "be", "Ruby"), ("Alice", "belong", "Amber"),
    }
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == "Bruno is Ruby."
    assert alignment.pred_sim == 1.0
    assert alignment.subject_fid == alignment.object_fid == alignment.rsf == 0.0
    assert result.hallucination_risk == 1.0


def test_no_predicate_match_remains_neutral_despite_matching_subject():
    result = stave_verify("Alice weighs 60 kilograms.", "Alice studied Physics.")
    assert len(result.answer_triplets) == len(result.knowledge_triplets) == 1
    assert result.knowledge_triplets[0].predicate == "study"
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet is None
    assert alignment.pred_sim == alignment.rsf == 0.0
    assert not alignment.wrong_slot
    assert result.n_matched == 0
    assert result.hallucination_risk == 0.5
    assert result.verdict == "pass"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice Smith weighs 60 kilograms.",
                                "Alice Jones weighs 80 kilograms."))),
)
def test_partial_alias_cannot_supply_another_owners_value(facts):
    result = stave_verify("Alice Smith weighs 80 kilograms.", " ".join(facts))
    assert {t.subject for t in result.knowledge_triplets} == {"Alice Smith", "Alice Jones"}
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.subject == "Alice Smith"
    assert alignment.knowledge_triplet.object_ == "60"
    assert alignment.subject_fid == 1.0
    assert alignment.object_fid == alignment.rsf == 0.0
    assert alignment.wrong_slot
    assert result.hallucination_risk == 1.0


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice weighs 60 kilograms.",
                                "Alice Smith has 80 books.",
                                "Alice is Alice Smith."))),
)
@pytest.mark.parametrize("verb", ("weighs", "WEIGHS"))
def test_alias_mass_uses_exact_surface_predicate_before_stronger_subject(facts, verb):
    # Declared alias: the mass answer is supported, and 80 counts books only.
    # Current extraction drops the alias sentence itself; this tests partial
    # subject fidelity, not semantic alias resolution. The expected 0.175 is
    # the existing formula: 1 - (0.35 * 0.5 + 0.65 * 1.0).
    answer = f"Alice Smith {verb} 60 kilograms."
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    assert {(t.subject, t.predicate, t.predicate_raw, t.object_)
            for t in result.knowledge_triplets} == {
        ("Alice", "have", "weighs", "60"),
        ("Alice Smith", "have", "has", "80"),
    }
    assert len(result.answer_triplets) == len(result.alignments) == 1
    alignment = result.alignments[0]
    assert alignment.answer_triplet.predicate_raw == verb
    assert alignment.knowledge_triplet.sentence == "Alice weighs 60 kilograms."
    assert alignment.pred_sim == alignment.object_fid == 1.0
    assert alignment.subject_fid == 0.5
    assert alignment.rsf == pytest.approx(0.825)
    assert not alignment.wrong_slot
    assert not result.wrong_slot_detected
    assert result.hallucination_risk == stave_risk(answer, passage) == 0.175
    assert result.verdict == "pass"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice weighs 60 kilograms.",
                                "Alice Smith has 80 books.",
                                "Alice is Alice Smith."))),
)
@pytest.mark.parametrize("person", ("Alice", "Alice Smith"))
def test_alias_wrong_mass_cannot_borrow_eighty_from_book_count(facts, person):
    # Fixed truth: both names denote the same person, whose mass is 60 kg.
    answer = f"{person} weighs 80 kilograms."
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == "Alice weighs 60 kilograms."
    assert alignment.knowledge_triplet.predicate_raw == "weighs"
    assert alignment.subject_fid == (1.0 if person == "Alice" else 0.5)
    assert alignment.object_fid == alignment.rsf == 0.0
    assert alignment.wrong_slot
    assert result.n_wrong_slot == 1
    assert result.hallucination_risk == stave_risk(answer, passage) == 1.0
    assert result.verdict == "flag"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice weighs 60 kilograms.",
                                "Bruno has 80 books."))),
)
@pytest.mark.parametrize("mass,expected_risk,wrong_slot", (("80", 0.5, False), ("60", 1.0, True)))
def test_unknown_mass_is_not_supported_by_same_subjects_book_count(
    facts, mass, expected_risk, wrong_slot,
):
    # Bruno's mass is unspecified, not contradicted. Neither a book count nor
    # Alice's mass supplies it, even if a number happens to match the answer.
    answer = f"Bruno weighs {mass} kilograms."
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    assert {(t.subject, t.predicate_raw, t.object_) for t in result.knowledge_triplets} == {
        ("Alice", "weighs", "60"), ("Bruno", "has", "80"),
    }
    alignment = result.alignments[0]
    assert alignment.pred_sim == 1.0
    assert alignment.subject_fid == alignment.rsf == 0.0
    assert alignment.object_fid == (1.0 if mass == "60" else 0.0)
    assert alignment.wrong_slot is wrong_slot
    assert result.wrong_slot_detected is wrong_slot
    assert result.hallucination_risk == stave_risk(answer, passage) == expected_risk
    if mass == "80":
        # A coherent book-count candidate is not proof of a mass assertion;
        # the unrelated exact verb is not grounds for a hard flag either.
        assert alignment.knowledge_triplet is None
        assert result.n_matched == 0
        assert result.verdict == "pass"
    else:
        assert alignment.knowledge_triplet.sentence == "Alice weighs 60 kilograms."
        assert result.n_matched == 1
        assert result.verdict == "flag"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Bruno cofounded Ruby.", "Alice co-founded Amber."))),
)
@pytest.mark.parametrize("verb", ("co-founded", "cofounded", "CO-FOUNDED"))
def test_hyphen_equivalent_predicates_preserve_supported_founder_in_both_orders(facts, verb):
    answer = f"Bruno {verb} Ruby."
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    assert {(t.subject, t.predicate, t.predicate_raw, t.object_)
            for t in result.knowledge_triplets} == {
        ("Bruno", "found", "cofounded", "Ruby"),
        ("Alice", "found", "co-founded", "Amber"),
    }
    assert len(result.answer_triplets) == len(result.alignments) == 1
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == "Bruno cofounded Ruby."
    assert alignment.pred_sim == alignment.subject_fid == alignment.object_fid == 1.0
    assert alignment.rsf == 1.0
    assert not alignment.wrong_slot
    assert not result.wrong_slot_detected
    assert result.hallucination_risk == stave_risk(answer, passage) == 0.0


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Bruno cofounded Ruby.",
                                "BRUNO CO-FOUNDED Ruby.",
                                "Bruno CoFounded Ruby."))),
)
def test_case_and_hyphen_duplicate_predicates_retain_first_surface(facts):
    answer = "Bruno co-founded Ruby."
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    assert len(result.knowledge_triplets) == 1
    kept = result.knowledge_triplets[0]
    assert kept.sentence == facts[0]
    assert kept.subject == facts[0].split()[0]
    assert kept.predicate_raw == facts[0].split()[1]
    assert kept.predicate == "found"
    assert kept.object_ == "Ruby"
    assert result.alignments[0].knowledge_triplet == kept
    assert not result.wrong_slot_detected
    assert result.hallucination_risk == stave_risk(answer, passage) == 0.0


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Bruno created Ruby.", "Alice founded Amber."))),
)
def test_coherent_different_raw_predicate_defers_instead_of_hard_flagging(facts):
    # These verbs share a broad canonical group, but the surface conflict
    # alone proves neither synonym entailment nor contradiction.
    answer = "Bruno founded Ruby."
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    assert {(t.subject, t.predicate, t.predicate_raw, t.object_)
            for t in result.knowledge_triplets} == {
        ("Bruno", "found", "created", "Ruby"),
        ("Alice", "found", "founded", "Amber"),
    }
    assert len(result.answer_triplets) == len(result.alignments) == 1
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet is None
    assert alignment.pred_sim == 1.0
    assert alignment.subject_fid == alignment.object_fid == alignment.rsf == 0.0
    assert not alignment.wrong_slot
    assert not result.wrong_slot_detected
    assert result.n_matched == result.n_wrong_slot == 0
    assert result.stave_score == 0.5
    assert result.hallucination_risk == stave_risk(answer, passage) == 0.5
    assert result.verdict == "pass"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice weighs 60 kilograms.",
                                "Alice has 60 books.",
                                "Bruno weighs 80 kilograms."))),
)
def test_canonical_collision_preserves_both_distinct_source_predicates(facts):
    result = stave_verify("Alice weighs 60 kilograms.", " ".join(facts))
    assert len(result.knowledge_triplets) == 3
    assert {(t.subject, t.predicate, t.predicate_raw, t.object_)
            for t in result.knowledge_triplets} == {
        ("Alice", "have", "weighs", "60"),
        ("Alice", "have", "has", "60"),
        ("Bruno", "have", "weighs", "80"),
    }
    assert {t.sentence for t in result.knowledge_triplets} == set(facts)


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice weighs 60 kilograms.",
                                "Alice has 60 books.",
                                "Bruno weighs 80 kilograms."))),
)
@pytest.mark.parametrize(
    "answer,raw_predicate",
    (("Alice weighs 60 kilograms.", "weighs"), ("Alice has 60 books.", "has")),
)
def test_supported_mass_and_book_count_survive_canonical_dedup_collision(
    facts, answer, raw_predicate,
):
    # Both claims are explicitly declared facts, regardless of sentence order.
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    assert len(result.knowledge_triplets) == 3
    assert len(result.answer_triplets) == len(result.alignments) == 1
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == answer
    assert alignment.knowledge_triplet.predicate_raw == raw_predicate
    assert alignment.pred_sim == alignment.subject_fid == alignment.object_fid == 1.0
    assert alignment.rsf == 1.0
    assert not alignment.wrong_slot
    assert not result.wrong_slot_detected
    assert result.hallucination_risk == stave_risk(answer, passage) == 0.0
    assert result.verdict == "pass"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice weighs 60 kilograms.",
                                "Alice has 60 books.",
                                "Bruno weighs 80 kilograms."))),
)
def test_wrong_mass_cannot_borrow_bruno_value_after_raw_predicate_dedup(facts):
    # Alice's declared mass is 60 kg; Bruno's 80 kg cannot supply her value.
    answer = "Alice weighs 80 kilograms."
    passage = " ".join(facts)
    result = stave_verify(answer, passage)
    assert len(result.knowledge_triplets) == 3
    alignment = result.alignments[0]
    assert alignment.knowledge_triplet.sentence == "Alice weighs 60 kilograms."
    assert alignment.knowledge_triplet.predicate_raw == "weighs"
    assert alignment.pred_sim == alignment.subject_fid == 1.0
    assert alignment.object_fid == alignment.rsf == 0.0
    assert alignment.wrong_slot
    assert result.n_wrong_slot == 1
    assert result.hallucination_risk == stave_risk(answer, passage) == 1.0
    assert result.verdict == "flag"


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice Smith owns Ruby Stone.",
                                "Alice Jones owns Ruby Gem."))),
)
def test_subject_priority_preserves_one_triplets_soft_fidelity_formula(facts):
    # Explicit triplets isolate the numeric alignment contract: the current
    # NP extractor truncates these adjacent two-word object names to "Ruby".
    by_sentence = {
        "Alice Smith owns Ruby Stone.": Triplet(
            "Alice Smith", "own", "owns", "Ruby Stone", "Alice Smith owns Ruby Stone.",
        ),
        "Alice Jones owns Ruby Gem.": Triplet(
            "Alice Jones", "own", "owns", "Ruby Gem", "Alice Jones owns Ruby Gem.",
        ),
    }
    answer = Triplet("Alice Smith", "own", "owns", "Ruby Gem", "Alice Smith owns Ruby Gem.")
    alignment = _align_one(answer, [by_sentence[fact] for fact in facts])
    assert alignment.knowledge_triplet.subject == "Alice Smith"
    assert alignment.knowledge_triplet.object_ == "Ruby Stone"
    assert alignment.subject_fid == 1.0
    assert alignment.object_fid == pytest.approx(1.0 / 3.0)
    assert alignment.rsf == pytest.approx(0.35 + 0.65 / 3.0)
    assert not alignment.wrong_slot


@pytest.mark.parametrize(
    "facts", tuple(permutations(("Alice Smith owns Amber.",
                                "Alice Smith owns Ruby."))),
)
def test_equal_subject_fidelity_uses_stronger_object_fidelity(facts):
    result = stave_verify("Alice Smith owns Ruby.", " ".join(facts))
    alignment = result.alignments[0]
    assert {t.object_ for t in result.knowledge_triplets} == {"Amber", "Ruby"}
    assert alignment.knowledge_triplet.sentence == "Alice Smith owns Ruby."
    assert alignment.subject_fid == alignment.object_fid == alignment.rsf == 1.0
    assert result.hallucination_risk == 0.0


def test_equal_predicate_and_slot_ranks_choose_stable_canonical_fact():
    facts = MASS_FACTS[1:] + ("Dorian weighs 77 kilograms.",)
    selected = set()
    for order in permutations(facts):
        result = stave_verify(MASS_FACTS[0], " ".join(order))
        alignment = result.alignments[0]
        assert alignment.pred_sim == 1.0
        assert alignment.subject_fid == alignment.object_fid == alignment.rsf == 0.0
        assert alignment.knowledge_triplet.sentence in facts
        assert result.hallucination_risk == 1.0
        selected.add((alignment.knowledge_triplet.subject,
                      alignment.knowledge_triplet.predicate,
                      alignment.knowledge_triplet.object_))
    assert len(selected) == 1


@pytest.mark.parametrize(
    "facts", (tuple(permutations((MASS_FACTS[0], "ALICE weighs 60 kilograms.")))
              + tuple(permutations((MASS_FACTS[0], "ALICE WEIGHS 60 kilograms.")))),
)
def test_identical_token_dedup_preserves_first_surface_provenance(facts):
    # Same canonical/token triples and casefolded raw predicate still dedup,
    # retaining the first source surface. Do not promise raw provenance equality.
    result = stave_verify(MASS_FACTS[0], " ".join(facts))
    assert len(result.knowledge_triplets) == 1
    assert result.knowledge_triplets[0].predicate_raw == facts[0].split()[1]
    assert result.knowledge_triplets[0].predicate_raw.casefold() == "weighs"
    assert result.knowledge_triplets[0].sentence == facts[0]
    assert result.alignments[0].knowledge_triplet.sentence == facts[0]
    assert result.hallucination_risk == 0.0


@pytest.fixture
def analyzer_pair(monkeypatch):
    monkeypatch.setenv("ENTROLY_STAVE", "1")
    monkeypatch.setenv("ENTROLY_LOCAL_NLI", "0")
    monkeypatch.setattr(witness, "_DEFAULT_RISK_MODEL", RiskModel())
    options = dict(use_nli=False, use_local_nli=False, force_python=True,
                   profile="benchmark_qa")
    return (WitnessAnalyzer(use_stave=False, **options),
            WitnessAnalyzer(use_stave=True, **options))


@pytest.mark.parametrize("facts", TWO_ORDERS)
@pytest.mark.parametrize(
    "answer,stave_expected",
    ((MASS_FACTS[0], 0.0), (MASS_FACTS[1], 0.0),
     ("Alice weighs 80 kilograms.", 1.0), ("Bruno weighs 60 kilograms.", 1.0)),
)
def test_real_witness_blend_uses_each_orders_own_base_without_changing_policy(
    facts, answer, stave_expected, analyzer_pair,
):
    without_stave, with_stave = analyzer_pair
    context = " ".join(facts)
    base, base_policy = without_stave.analyze_and_rewrite(context, answer, mode="strict")
    combined, combined_policy = with_stave.analyze_and_rewrite(context, answer, mode="strict")
    assert with_stave.use_stave
    assert not with_stave.use_nli and not with_stave.use_local_nli
    assert with_stave.force_python
    assert stave_risk(answer, context) == stave_expected
    expected = (0.75 * base.summary_score + 0.25 if stave_expected == 0.0
                else min(base.summary_score, 0.10))
    assert combined.summary_score == pytest.approx(expected)
    assert len(base.certificates) == len(combined.certificates) == 1
    assert combined.certificates == base.certificates
    assert combined_policy == base_policy
    assert with_stave.nli_calls == without_stave.nli_calls == 0


@pytest.mark.parametrize("facts", TWO_ORDERS)
@pytest.mark.parametrize(
    "answer,expected_score",
    ((MASS_FACTS[0], 0.70), (MASS_FACTS[1], 0.70),
     ("Alice weighs 80 kilograms.", 0.10), ("Bruno weighs 60 kilograms.", 0.10)),
)
def test_fixed_base_isolates_stave_score_and_preserves_strict_certificate_policy(
    facts, answer, expected_score, analyzer_pair, monkeypatch,
):
    def fixed_certificate(claim, context):
        return Certificate(
            claim_id=claim.id, claim_text=claim.text, label="unknown",
            support_strength=0.5, contradiction_strength=0.0,
            evidence_adequacy=0.8, risk=0.4,
            proof_path=[ProofStep("test_fixed_base", "fixed scalar, no model", 0.4)],
        )

    without_stave, with_stave = analyzer_pair
    monkeypatch.setattr(without_stave, "_certify_claim", fixed_certificate)
    monkeypatch.setattr(with_stave, "_certify_claim", fixed_certificate)
    context = " ".join(facts)
    base, base_policy = without_stave.analyze_and_rewrite(context, answer, mode="strict")
    result, policy = with_stave.analyze_and_rewrite(context, answer, mode="strict")
    assert base.summary_score == pytest.approx(0.6)
    assert result.summary_score == pytest.approx(expected_score)
    assert len(result.certificates) == 1
    assert result.certificates[0].risk == 0.4
    assert result.certificates == base.certificates
    assert policy == base_policy
    assert policy.mode == "strict" and policy.profile == "benchmark_qa"
    assert policy.flagged_count == policy.warned_count == 1
    assert policy.suppressed_count == 0
    assert "unverified" in policy.output
    assert with_stave.nli_calls == without_stave.nli_calls == 0
