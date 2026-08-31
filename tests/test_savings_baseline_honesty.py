"""A saving is measured against a prompt someone could have sent.

An earlier dogfooding pass fixed half of this. It found that `tokens_saved` was
`corpus - selected`, so a query matching nothing reported the largest saving,
and it credited zero whenever the selection was not evidence-backed. That
contract lives in test_no_match_honesty.py.

The other half survived: an evidence-backed selection still billed the entire
corpus. Measured on this repository, one `optimize_context` call reported
4,919,764 tokens saved against 2,941 actually sent -- a ratio of 1,672 to 1 --
and the whole suite stayed green, because every existing test asserted on the
evidence gate and none on the baseline.

Nobody pastes a five-million-token repository into a model; it exceeds every
context window. Crediting it measures against a counterfactual that cannot
happen, and makes the reported figure scale with repository size rather than
with anything Entroly did.

`cli.py` had already settled this question in four places, with a comment
reading "Claiming savings vs. the whole repo (7M+ tokens) is marketing, not
measurement." The engine -- the surface that runs in production -- was the one
not applying it.
"""

from __future__ import annotations

import pathlib
import re

from entroly.engine import naive_context_baseline


class TestBaseline:
    def test_a_large_repository_is_capped_to_a_plausible_prompt(self):
        assert naive_context_baseline(4_922_705) == 32_000

    def test_a_small_project_is_never_credited_with_more_than_it_holds(self):
        assert naive_context_baseline(5_000) == 5_000
        assert naive_context_baseline(0) == 0

    def test_a_degenerate_input_never_yields_a_negative_baseline(self):
        for bad in (None, -1, 0, ""):
            assert naive_context_baseline(bad) >= 0

    def test_the_engine_and_the_cli_state_the_same_standard(self):
        """One product, one baseline. These disagreed for the shipping surface."""
        cli = pathlib.Path("entroly/cli.py").read_text(encoding="utf-8")
        assert re.search(r"min\(\s*\w*total\w*\s*,\s*32_?000\s*\)", cli), (
            "cli.py no longer states a naive baseline; keep the two in step"
        )
        assert naive_context_baseline(10**9) == 32_000


class TestAppliedWhereItMatters:
    def test_both_engine_call_sites_measure_against_the_baseline(self):
        """A saving computed from the raw corpus is the defect this file exists for."""
        engine = pathlib.Path("entroly/engine.py").read_text(encoding="utf-8")
        raw = re.findall(
            r"_honest_tokens_saved\(\s*[^)]*?total_available_tokens\s*-", engine, re.S
        )
        assert not raw, (
            "a call site is still passing `corpus - selected`; wrap the total in "
            "naive_context_baseline() so the saving is measured against a prompt "
            "a caller could plausibly have sent"
        )

    def test_the_reported_saving_stays_within_the_baseline(self):
        """The ceiling that makes the figure defensible, stated as an assertion."""
        for corpus, used in ((4_922_705, 2_941), (10_000, 1_000), (500, 100)):
            saving = naive_context_baseline(corpus) - used
            assert saving <= naive_context_baseline(corpus)
            assert saving <= max(corpus, 0), "cannot save more than exists"
