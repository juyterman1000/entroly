"""Selection must not fabricate relevance, and savings must not count withholding.

Dogfooding found that a query matching nothing still produced a ranked list:
score normalization floors and uniform fallbacks give every fragment a middling
score, so `optimize_context` returned confident-looking unrelated files and then
billed the omitted corpus as "tokens saved". Measured on this repository, the
nonsense query "zzqqxx blorptastic wubbleflux" selected 8 files at ~0.6
relevance and shared its top hits with a real query, while `tokens_saved` stayed
~2.65M regardless of query — it was simply `corpus - selected`.

The contract these tests pin:
  * unrelated context is worse than none -> report `no_match`
  * withholding context because nothing matched is not a saving -> credit zero
  * pinned/required evidence is operator policy, not relevance -> always kept
"""

from __future__ import annotations

from entroly.server import _evidence_backed, _honest_tokens_saved


def test_positive_relevance_is_evidence_backed():
    assert _evidence_backed([{"source": "a.py", "relevance": 0.62}]) is True
    assert _evidence_backed([{"source": "a.py", "relevance_score": 1.4}]) is True


def test_zero_relevance_is_not_evidence_backed():
    assert _evidence_backed([{"source": "a.py", "relevance": 0.0}]) is False
    assert _evidence_backed([{"source": "a.py", "relevance_score": 0}]) is False
    assert _evidence_backed([]) is False


def test_pinned_fragments_alone_are_not_evidence():
    # Pinned content is included by operator policy, not because it matched.
    # Counting it would make every no-match look matched.
    pinned_only = [{"source": "always.md", "is_pinned": True, "relevance": 0.9}]
    assert _evidence_backed(pinned_only) is False
    mixed = pinned_only + [{"source": "hit.py", "relevance": 0.3}]
    assert _evidence_backed(mixed) is True


def test_malformed_relevance_never_counts_as_evidence():
    for bad in (None, "", "not-a-number", [], {}):
        assert _evidence_backed([{"source": "a.py", "relevance": bad}]) is False


def test_savings_are_zero_when_nothing_matched():
    # The core accounting defect: the WORSE the match, the LARGER the reported
    # saving, because `corpus - selected` grows as selection shrinks.
    assert _honest_tokens_saved([{"relevance": 0.0}], 2_654_766) == 0
    assert _honest_tokens_saved([], 2_654_766) == 0
    assert _honest_tokens_saved(
        [{"is_pinned": True, "relevance": 1.0}], 2_654_766
    ) == 0


def test_savings_are_credited_when_evidence_was_found():
    assert _honest_tokens_saved([{"relevance": 0.62}], 5000) == 5000


def test_savings_are_never_negative():
    assert _honest_tokens_saved([{"relevance": 0.62}], -17) == 0
    assert _honest_tokens_saved([{"relevance": 0.62}], None) == 0


# ── Lexical verification (independent of the ranker's scores) ────────────────
# Relevance scores cannot detect no-match: the native ranker floors and
# normalizes, so a query matching nothing returns ~0.6. Checking the delivered
# text is the one signal the scoring pipeline cannot fabricate.

def test_selection_matching_requires_a_query_term_in_the_delivered_text():
    from entroly.server import _selection_matches_query

    hit = [{"source": "entroly/proxy.py", "content": "def inject_context(): ..."}]
    assert _selection_matches_query("inject context proxy", hit) is True
    # High score, zero lexical overlap -> the fabricated-relevance case.
    miss = [{"source": "docs/readme.md", "content": "unrelated prose",
             "relevance": 0.6}]
    assert _selection_matches_query("zzqqxx blorptastic wubbleflux", miss) is False


def test_pinned_content_cannot_satisfy_the_query_match():
    from entroly.server import _selection_matches_query

    pinned = [{"source": "always.md", "content": "inject context proxy",
               "is_pinned": True}]
    assert _selection_matches_query("inject context proxy", pinned) is False


def test_stopword_only_query_is_not_treated_as_a_no_match():
    from entroly.server import _selection_matches_query

    # No lexical intent to satisfy; must not flip everything to no_match.
    assert _selection_matches_query("what is the", [{"content": "anything"}]) is True
    assert _selection_matches_query("", [{"content": "anything"}]) is True


def test_source_path_counts_as_evidence():
    from entroly.server import _selection_matches_query

    assert _selection_matches_query(
        "checkpoint", [{"source": "entroly/checkpoint.py", "content": ""}]
    ) is True


# ── Degenerate-ranking detection (variance, not magnitude) ───────────────────
# A ranker that found evidence separates candidates; one that found none cannot.
# Measured: real query -> 0.8534/0.7220/0.7095/0.6971 (spread 0.156);
# no-match -> 0.0800 x4 (spread 0.000). Thresholding the score never worked
# because 0.08 and 0.62 are both "positive"; only one of them is a ranking.

def test_flat_scores_are_detected_as_no_ranking():
    from entroly.server import _score_distribution_is_degenerate

    flat = [{"relevance": 0.08} for _ in range(4)]
    assert _score_distribution_is_degenerate(flat) is True


def test_separated_scores_are_a_real_ranking():
    from entroly.server import _score_distribution_is_degenerate

    real = [{"relevance": v} for v in (0.8534, 0.7220, 0.7095, 0.6971)]
    assert _score_distribution_is_degenerate(real) is False


def test_high_but_flat_scores_are_still_no_ranking():
    from entroly.server import _score_distribution_is_degenerate

    # Magnitude is not evidence: uniformly high is as uninformative as
    # uniformly low.
    assert _score_distribution_is_degenerate(
        [{"relevance": 0.95} for _ in range(5)]
    ) is True


def test_single_candidate_is_not_judged_by_variance():
    from entroly.server import _score_distribution_is_degenerate

    # One result can be neither flat nor spread; the lexical check decides.
    assert _score_distribution_is_degenerate([{"relevance": 0.08}]) is False


def test_pinned_scores_do_not_create_false_spread():
    from entroly.server import _score_distribution_is_degenerate

    salted = [{"relevance": 0.9, "is_pinned": True}] + [
        {"relevance": 0.08} for _ in range(3)
    ]
    assert _score_distribution_is_degenerate(salted) is True


# ── the contract itself, and the wiring that applies it ──────────────────────
#
# Every test above this line checks a *predicate* in isolation. None of them
# checked that anything calls the contract, and that is exactly how the bug
# below survived: `_evidence_backed`, `_selection_matches_query` and
# `_score_distribution_is_degenerate` were all correct and all tested, while the
# guard composed from them lived only in the MCP tool handler nested inside
# `create_mcp_server`. `EntrolyEngine.optimize_context` -- the method the CLI,
# the SDK and the proxy every one of them call -- had no guard at all, so a
# query matching nothing returned confident unrelated files everywhere but MCP.
#
# Correct parts, unwired. The tests below pin the composition and the wiring.

def _selection(relevance, text="def handler(session_token): return validate(token)"):
    """A selection whose scores are whatever the caller says they are.

    Pass a sequence for a real ranking, a scalar for a flat one -- the
    difference is load-bearing, since a flat distribution is itself the
    degeneracy signal.
    """
    scores = relevance if isinstance(relevance, (list, tuple)) else [relevance] * 4
    return [
        {"source": f"file:mod_{i}.py", "content": text, "relevance": score,
         "token_count": 40}
        for i, score in enumerate(scores)
    ]


def test_contract_wipes_to_pinned_and_says_why():
    from entroly.server import apply_no_match_contract

    result = {
        "selected_fragments": _selection(0.0, text="unrelated cache economics"),
        "total_tokens": 160,
        "tokens_saved": 99_999,
        "total_fragments": 12,
    }

    apply_no_match_contract(result, "zzqqxx blorptastic wubbleflux")

    assert result["status"] == "no_match"
    assert result["selected_fragments"] == []
    # Withholding context because nothing matched is not a saving.
    assert result["tokens_saved"] == 0
    assert result["total_tokens"] == 0
    assert result["no_match"]["candidates_considered"] == 12
    assert result["no_match"]["remediation"]


def test_contract_keeps_pinned_evidence_through_a_no_match():
    """Pinned fragments are operator policy, not relevance."""
    from entroly.server import apply_no_match_contract

    pinned = {"source": "file:policy.py", "content": "x", "is_pinned": True,
              "token_count": 7}
    result = {"selected_fragments": _selection(0.0, text="unrelated") + [pinned]}

    apply_no_match_contract(result, "zzqqxx blorptastic wubbleflux")

    assert result["selected_fragments"] == [pinned]
    assert result["no_match"]["pinned_retained"] == 1
    assert result["tokens_used"] == 7


def test_contract_leaves_a_real_match_alone():
    from entroly.server import apply_no_match_contract

    # A real ranking separates its candidates; a flat one would trip the
    # variance discriminator, which is the point of that discriminator.
    result = {
        "selected_fragments": _selection([0.91, 0.74, 0.66, 0.52]),
        "tokens_saved": 1_234,
    }

    apply_no_match_contract(result, "validate session token")

    assert "status" not in result
    assert len(result["selected_fragments"]) == 4
    assert result["tokens_saved"] == 1_234


def test_flat_indicator_scores_do_not_trip_when_they_are_not_a_ranking():
    """QCCR relevance is a match indicator, not a rank.

    `_rust_select` returns synthetic per-file fragments scored uniformly -- 1.0
    when the query matched, 0.0 when it did not. A spread test reads every one
    of those as "degenerate", so leaving the variance discriminator on for that
    caller would discard every successful selection. Measured, not assumed.
    """
    from entroly.server import apply_no_match_contract

    matched = {"selected_fragments": _selection(1.0)}
    apply_no_match_contract(matched, "validate session token",
                            scores_are_ranked=False)
    assert "status" not in matched

    # ...and the same shape with the variance test on would be wiped, which is
    # why the caller has to declare what its scores mean.
    wiped = {"selected_fragments": _selection(1.0)}
    apply_no_match_contract(wiped, "validate session token",
                            scores_are_ranked=True)
    assert wiped["status"] == "no_match"


def _calls_the_contract(function_node) -> bool:
    import ast

    return any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "apply_no_match_contract"
        for node in ast.walk(function_node)
    )


def _find_function(qualifier: str, name: str, module: str = "entroly/server.py"):
    """Locate a function by its enclosing scope, in whichever module holds it.

    The module is a parameter because `EntrolyEngine` moved to
    `entroly/engine.py` while `create_mcp_server` stayed in `entroly/server.py`.
    Hard-coding one file made these wiring tests fail the moment the engine was
    separated from the server -- correctly, since the guard they check had moved
    with it.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(module).read_text(encoding="utf-8", errors="replace"))
    for outer in ast.walk(tree):
        if getattr(outer, "name", None) != qualifier:
            continue
        for inner in ast.walk(outer):
            if (
                isinstance(inner, ast.FunctionDef)
                and inner.name == name
                and inner is not outer
            ):
                return inner
    raise AssertionError(f"{qualifier}.{name} not found in {module}")


def test_engine_optimize_context_is_wired_to_the_contract():
    """The regression that matters: the guard must be *called*, not just exist.

    `EntrolyEngine.optimize_context` is the method the CLI, SDK and proxy use.
    It went without this guard while a same-named MCP handler had it.
    """
    assert _calls_the_contract(
        _find_function("EntrolyEngine", "optimize_context", "entroly/engine.py")
    )


def test_mcp_tool_handler_is_wired_to_the_contract():
    """The MCP surface keeps its guard as the engine gains one."""
    assert _calls_the_contract(_find_function("create_mcp_server", "optimize_context"))


# ── unmeasured relevance is not zero relevance ───────────────────────────────
#
# Relevance is computed during selection and never stored on a fragment, so a
# selection that did not come from a ranker arrives without one. The fast path
# is exactly that: it replays a crystallized recipe out of the fragment store,
# and those fragments carry id/source/content/token_count and no score.
#
# Reading "never scored" as "scored zero" would convict every fast-path hit --
# a selection promoted *because* its fitness was measured -- of being unrelated.
# The sufficiency certificate settled this argument once already by reporting
# `boundary_exposure_measured=False` instead of a zero it could not compute.

def _unscored(text="def login(user): return authenticate(user)"):
    """A replayed selection: store shape, no relevance field."""
    return [
        {"id": f"frag-{i}", "source": f"file:auth_{i}.py", "content": text,
         "token_count": 30}
        for i in range(3)
    ]


def test_evidence_signal_separates_absent_from_zero():
    from entroly.server import _evidence_signal

    assert _evidence_signal(_selection(0.7)) is True
    assert _evidence_signal(_selection(0.0)) is False
    assert _evidence_signal(_unscored()) is None
    # Pinned-only carries no measurable evidence either way.
    assert _evidence_signal([{"source": "p.py", "relevance": 0.9,
                              "is_pinned": True}]) is None


def test_evidence_backed_keeps_its_two_valued_contract():
    """The bool view must not shift under callers that only ask yes/no."""
    from entroly.server import _evidence_backed

    assert _evidence_backed(_selection(0.7)) is True
    assert _evidence_backed(_selection(0.0)) is False
    assert _evidence_backed(_unscored()) is False
    assert _evidence_backed([]) is False


def test_unscored_selection_survives_when_the_text_answers_the_query():
    """A missing field must not convict a selection nothing ever scored."""
    from entroly.server import apply_no_match_contract

    result = {"selected_fragments": _unscored()}

    apply_no_match_contract(result, "how does login authenticate",
                            scores_are_ranked=False)

    assert "status" not in result
    assert len(result["selected_fragments"]) == 3


def test_unscored_selection_still_trips_when_it_shares_nothing():
    """Unmeasured relevance abstains; it does not grant immunity.

    A regex-triggered skill that fires too broadly replays a recipe unrelated to
    what was asked. With no score to consult, the lexical check has to decide --
    and it must still be able to say no.
    """
    from entroly.server import apply_no_match_contract

    result = {"selected_fragments": _unscored(text="unrelated cache economics")}

    apply_no_match_contract(result, "zzqqxx blorptastic wubbleflux",
                            scores_are_ranked=False)

    assert result["status"] == "no_match"
    assert result["no_match"]["evidence_measured"] is False
    assert result["no_match"]["lexical_match"] is False


def test_payload_reports_which_signal_convicted():
    from entroly.server import apply_no_match_contract

    scored = {"selected_fragments": _selection(0.0, text="unrelated cache")}
    apply_no_match_contract(scored, "zzqqxx blorptastic")

    assert scored["no_match"]["evidence_measured"] is True


def test_fast_path_return_is_wired_to_the_contract():
    """The fast path exits before the pipeline, so it needs its own call.

    A promoted skill is fitness-gated and freshness-checked, but it is triggered
    by a regex over the query; a pattern that fires too broadly replays a recipe
    that has nothing to do with the question.
    """
    import ast

    engine_method = _find_function(
        "EntrolyEngine", "optimize_context", "entroly/engine.py"
    )
    for node in ast.walk(engine_method):
        if not isinstance(node, ast.If):
            continue
        source = ast.dump(node)
        if "fp_result" in source and "apply_no_match_contract" in source:
            return
    raise AssertionError("fast-path early return does not apply the contract")


# ── morphology: token membership, not substring containment ──────────────────
#
# The lexical check is the one signal the scoring pipeline cannot fabricate, so
# it decides alone whenever relevance is unmeasured. That makes its precision
# load-bearing. Raw substring containment fails on ordinary English morphology
# in the direction that discards good work: it reports "no term in common" for
# a selection that answers the question.
#
# Found by `test_simulate_small_project`: the query "How are credit cards
# charged?" against `charge_card(customer.card, amount)` matched nothing --
# "cards" is not a substring of "card", "charged" is not a substring of
# "charge" -- so a correct selection was discarded as a no-match the moment the
# contract was wired into the engine method.

def test_plural_query_term_matches_singular_source_token():
    from entroly.server import _selection_matches_query

    billing = [{
        "source": "src/billing.py",
        "content": "def charge_card(customer, amount):\n"
                   "    return StripeGateway().charge(customer.card, amount)\n",
    }]
    assert _selection_matches_query("How are credit cards charged?", billing) is True


def test_past_tense_query_term_matches_present_tense_source_token():
    from entroly.server import _selection_matches_query

    frags = [{"source": "a.py", "content": "def verify_password(user, pw): ..."}]
    assert _selection_matches_query("who verifies the password", frags) is True


def test_stemming_does_not_make_unrelated_words_match():
    """Recall must not be bought with precision.

    A matcher loose enough to call anything a match would silently retire the
    guard -- the same outcome as not running it, which is what this whole
    contract exists to prevent.
    """
    from entroly.server import _selection_matches_query

    frags = [{"source": "src/billing.py",
              "content": "def charge_card(customer, amount): ..."}]
    assert _selection_matches_query(
        "zzqqxx blorptastic wubbleflux", frags
    ) is False
    # "discharge" must not match "charge" -- stems, not substrings.
    assert _selection_matches_query("discharge the capacitor", frags) is False


# ── the guard only judges an actual choice ───────────────────────────────────

def test_selection_of_every_candidate_is_never_a_no_match():
    """Nothing excluded means no selection decision to be wrong about.

    "Unrelated context is worse than none" is a claim about *displacing*
    relevant evidence under a budget. When the optimizer returned everything it
    had, nothing was displaced and withholding is pure loss.

    Measured: a two-file, 47-token repository against an 8,000-token budget,
    asked "How does the authentication flow work?", returned both files and was
    then wiped to zero -- `stem("authentication")` is None and never reaches the
    token `auth`, so the whole corpus was discarded over a vocabulary gap that
    no amount of discarding could relieve.
    """
    from entroly.server import apply_no_match_contract

    result = {
        "selected_fragments": _selection(0.0, text="unrelated cache economics"),
        "total_fragments": 4,
    }

    apply_no_match_contract(result, "zzqqxx blorptastic wubbleflux")

    assert "status" not in result
    assert len(result["selected_fragments"]) == 4


def test_the_guard_still_judges_a_real_subset():
    """A selection drawn from a wider corpus is a decision, and can be wrong."""
    from entroly.server import apply_no_match_contract

    result = {
        "selected_fragments": _selection(0.0, text="unrelated cache economics"),
        "total_fragments": 140,
    }

    apply_no_match_contract(result, "zzqqxx blorptastic wubbleflux")

    assert result["status"] == "no_match"


# ── query and haystack must share a tokenizer ────────────────────────────────

def test_snake_case_identifier_matches_the_code_that_defines_it():
    """The most precise thing a developer can type must not fail to match.

    `_query_terms` keeps `parse_manifest` whole; `_lexical_terms` splits the
    haystack into `parse`/`manifest`. Comparing across two tokenizers made
    every snake_case identifier a no-match. Substring matching hid this;
    token membership cannot, so the query is tokenised the same way.
    """
    from entroly.server import _selection_matches_query

    frags = [{
        "source": "file:parser.py",
        "content": "def parse_manifest(path):\n    return decode_manifest(path)\n",
    }]
    assert _selection_matches_query("parse_manifest decode_manifest", frags) is True


def test_tokenised_query_does_not_match_unrelated_code():
    from entroly.server import _selection_matches_query

    frags = [{"source": "file:billing.py",
              "content": "def charge_card(customer, amount): ..."}]
    assert _selection_matches_query("parse_manifest decode_manifest", frags) is False
