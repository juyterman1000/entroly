"""Deterministic unit tests for the Tier-0 zero-dep localizer.

No network, no model - pure logic. These pin the signal mechanics so a
regression can't silently degrade the localizer the way the SimHash
recall() bug silently degraded retrieval.
"""

from __future__ import annotations

from entroly.file_localizer import localize_fragments
from entroly.localization import Tier0Localizer, _module_to_paths, _split_ident
from entroly.server import EntrolyEngine

MINI_REPO = {
    "pkg/__init__.py": "",
    "pkg/config.py": (
        "import os\n"
        "def parse_config(path):\n"
        "    '''Read a config file.'''\n"
        "    return {}\n"
    ),
    "pkg/server.py": (
        "from pkg.config import parse_config\n"
        "def start_server():\n"
        "    cfg = parse_config('app.cfg')\n"
        "    return cfg\n"
    ),
    "pkg/utils.py": "def helper():\n    return 1\n",
    "tests/test_server.py": "def test_x():\n    assert True\n",
}


def test_split_ident():
    assert _split_ident("parse_config") == ["parse", "config"]
    assert _split_ident("HTTPServerError") == ["http", "server", "error"]


def test_module_to_paths():
    assert _module_to_paths("pkg.config") == [
        "pkg/config.py", "pkg/config/__init__.py"]


def test_symbol_table_and_import_graph_built():
    loc = Tier0Localizer(MINI_REPO)
    assert "pkg/config.py" in loc.sym_def["parse_config"]
    assert "pkg/config.py" in loc.imports["pkg/server.py"]
    assert "pkg/server.py" in loc.imported_by["pkg/config.py"]


def test_s1_extracts_traceback_path():
    loc = Tier0Localizer(MINI_REPO)
    issue = (
        'Traceback (most recent call last):\n'
        '  File "pkg/config.py", line 3, in parse_config\n'
        'KeyError\n'
    )
    assert loc._s1(issue)[0] == "pkg/config.py"


def test_s2_structure_expansion_finds_definer_and_neighbour():
    loc = Tier0Localizer(MINI_REPO)
    ranked = loc._s2("`parse_config` returns an empty dict unexpectedly")
    assert ranked, "S2 produced nothing for a defined symbol"
    assert ranked[0] == "pkg/config.py"
    assert "pkg/server.py" in ranked  # imported_by expansion


def test_rank_puts_relevant_file_first_and_is_total():
    loc = Tier0Localizer(MINI_REPO)
    issue = "parse_config in pkg/config.py crashes on missing file"
    ranked = loc.rank(issue, k=20)
    assert ranked[0] == "pkg/config.py"
    assert set(ranked) == set(MINI_REPO)
    assert len(ranked) == len(set(ranked))


def test_floor_is_bm25_when_no_structure_signal():
    loc = Tier0Localizer(MINI_REPO)
    ranked = loc.rank("helper function returning one", k=5)
    assert ranked[0] == "pkg/utils.py"


def test_pathological_issue_does_not_crash():
    loc = Tier0Localizer(MINI_REPO)
    for bad in ("", "```\n\n```", "::::....", "0 AND OR ((:::***"):
        out = loc.rank(bad, k=5)
        assert isinstance(out, list)


# ────────────────────────────────────────────────────────────────────
# engine_s6 — deterministic edit-target rerank
# These pin the FOUR structural guards (window, frozen cues, class
# re-prio with intent guards, test→source mirror). The discipline is
# the same as the SimHash-recall regression: a future refactor must
# either keep these invariants or explicitly retire the experiment.
# ────────────────────────────────────────────────────────────────────

EDIT_REPO = {
    # source
    "pkg/auth.py":      "def login():\n    return True\n",
    "pkg/cache.py":     "def lookup():\n    pass\n",
    "pkg/router.py":    "def route():\n    pass\n",
    "pkg/util.py":      "def helper():\n    pass\n",
    # tests
    "tests/test_auth.py":  "def test_login():\n    assert True\n",
    "tests/test_cache.py": "def test_lookup():\n    assert True\n",
    # non-source distractors
    "docs/index.rst":   "Welcome to the docs.\n",
    "docs/intro.md":    "# Intro\n",
    "HISTORY.rst":      "0.1.0 first release\n",
    "CHANGELOG.md":     "## 0.1.0\n",
    ".github/ISSUE_TEMPLATE/bug_report.yml": "name: Bug\n",
}


def test_edit_target_empty_input_returns_empty():
    loc = Tier0Localizer(EDIT_REPO)
    assert loc.rerank_edit_target([], "anything") == []


def test_edit_target_source_outranks_non_source_in_window():
    """No intent words, no explicit cue → docs/HISTORY/yml demoted
    BELOW any source file inside the top-20 window."""
    loc = Tier0Localizer(EDIT_REPO)
    base = [
        "docs/index.rst",                          # non-source
        ".github/ISSUE_TEMPLATE/bug_report.yml",   # non-source
        "HISTORY.rst",                             # non-source
        "pkg/auth.py",                             # source
        "pkg/util.py",                             # source
    ]
    out = loc.rerank_edit_target(base, "the login flow is broken", k=20)
    # every source must precede every non-source in the result
    src_pos = [out.index(p) for p in ("pkg/auth.py", "pkg/util.py")]
    ns_pos = [out.index(p) for p in (
        "docs/index.rst", "HISTORY.rst",
        ".github/ISSUE_TEMPLATE/bug_report.yml")]
    assert max(src_pos) < min(ns_pos)
    # permutation: every base entry preserved
    assert set(out) >= set(base)


def test_edit_target_test_demoted_below_source_no_test_intent():
    loc = Tier0Localizer(EDIT_REPO)
    base = ["tests/test_cache.py", "pkg/util.py"]
    out = loc.rerank_edit_target(base, "router crashes on empty input")
    assert out.index("pkg/util.py") < out.index("tests/test_cache.py")


def test_edit_target_doc_intent_guard_keeps_docs():
    """When the issue carries doc intent, non-source is NOT demoted."""
    loc = Tier0Localizer(EDIT_REPO)
    base = ["docs/index.rst", "pkg/util.py"]
    # "documentation" triggers _DOC_INTENT_WORDS → cls(non-source)=1
    issue = "the documentation is misleading and needs updating"
    out = loc.rerank_edit_target(base, issue)
    # tied class → original positions preserved
    assert out.index("docs/index.rst") < out.index("pkg/util.py")


def test_edit_target_test_intent_guard_keeps_tests():
    """When the issue carries test intent, tests are NOT demoted."""
    loc = Tier0Localizer(EDIT_REPO)
    base = ["tests/test_auth.py", "pkg/util.py"]
    issue = "the failing test does not catch the regression"
    out = loc.rerank_edit_target(base, issue)
    assert out.index("tests/test_auth.py") < out.index("pkg/util.py")


def test_edit_target_explicit_path_cue_frozen_first():
    """An exact path mention is an S1 cue → frozen at rank 0."""
    loc = Tier0Localizer(EDIT_REPO)
    base = ["docs/index.rst", "pkg/auth.py", "pkg/router.py"]
    issue = "see pkg/router.py — route() never returns"
    out = loc.rerank_edit_target(base, issue)
    assert out[0] == "pkg/router.py"


def test_edit_target_cue_order_follows_extraction_order():
    loc = Tier0Localizer(EDIT_REPO)
    base = ["pkg/util.py", "pkg/auth.py", "pkg/router.py"]
    # router mentioned BEFORE auth in the issue
    issue = "first pkg/router.py then pkg/auth.py fail"
    out = loc.rerank_edit_target(base, issue)
    assert out[:2] == ["pkg/router.py", "pkg/auth.py"]


def test_edit_target_test_to_source_mirror_promotion():
    """A test file in-window pulls its basename-mirror source in
    right after it — even if the mirror wasn't in base_ranked."""
    loc = Tier0Localizer(EDIT_REPO)
    # pkg/auth.py absent from base; mirror must be inserted
    base = ["tests/test_auth.py", "pkg/util.py"]
    out = loc.rerank_edit_target(base, "auth logic is wrong")
    # without test intent: util (source) first, then test, then mirror
    assert out.index("pkg/util.py") < out.index("tests/test_auth.py")
    assert "pkg/auth.py" in out
    assert (out.index("pkg/auth.py")
            == out.index("tests/test_auth.py") + 1)


def test_edit_target_window_guard_tail_untouched():
    """Files at base-rank > 20 are preserved in their base order
    (recall floor). The window is the metric horizon, NOT a tuned
    selectivity threshold."""
    loc = Tier0Localizer(EDIT_REPO)
    # 22 entries — last 2 are tail; first 20 are window
    base = (["pkg/auth.py", "pkg/cache.py", "pkg/router.py",
             "pkg/util.py",
             "tests/test_auth.py", "tests/test_cache.py",
             "docs/index.rst", "docs/intro.md",
             "HISTORY.rst", "CHANGELOG.md",
             ".github/ISSUE_TEMPLATE/bug_report.yml"]
            + ["pkg/auth.py"] * 0)               # filler intentionally none
    # pad with deterministic distinct synthetic paths so we cross 20
    pad = [f"pkg/_pad_{i}.py" for i in range(11)]
    base = base + pad        # len == 22; window=20, tail=2
    out = loc.rerank_edit_target(base, "router crash", k=50)
    # The two tail files keep their *relative* order at the end of
    # the produced ranking (recall floor).
    tail_files = base[20:]
    tail_positions = [out.index(f) for f in tail_files]
    assert tail_positions == sorted(tail_positions)
    # totality: nothing from the base dropped
    assert set(out) >= set(base)


def test_edit_target_pathological_issue_does_not_crash():
    loc = Tier0Localizer(EDIT_REPO)
    base = ["pkg/auth.py", "docs/index.rst", "tests/test_auth.py"]
    for bad in ("", "```\n```", "::::....", "0 AND OR ((:::***",
                "\x00\x01\x02"):
        out = loc.rerank_edit_target(base, bad)
        assert isinstance(out, list)
        # recall-safe even on garbage input
        assert set(out) >= set(base)


def test_edit_target_is_deterministic():
    """Same inputs → byte-identical output across calls. This pins
    the 'no hidden randomness' contract."""
    loc = Tier0Localizer(EDIT_REPO)
    base = ["docs/index.rst", "pkg/auth.py", "tests/test_auth.py",
            "pkg/util.py", "HISTORY.rst"]
    issue = "login broken when cache empty"
    a = loc.rerank_edit_target(base, issue)
    b = loc.rerank_edit_target(base, issue)
    assert a == b


def test_edit_target_no_intent_words_keep_doc_demotion():
    """Sanity: the doc-intent set must NOT trigger on common English
    that merely mentions a feature. 'login' must not unlock doc
    intent the way 'documentation' does."""
    loc = Tier0Localizer(EDIT_REPO)
    base = ["HISTORY.rst", "pkg/auth.py"]
    out = loc.rerank_edit_target(base, "login is broken")
    # source ahead of non-source
    assert out.index("pkg/auth.py") < out.index("HISTORY.rst")


def test_localize_fragments_reorders_sources_without_dropping_fragments():
    fragments = [
        {"source": "docs/auth.rst", "content": "login docs", "token_count": 10},
        {"source": "tests/test_auth.py", "content": "def test_login(): pass", "token_count": 10},
        {"source": "pkg/util.py", "content": "def helper(): pass", "token_count": 10},
        {"source": "", "content": "unkeyed", "token_count": 1},
    ]

    out = localize_fragments(fragments, "login is broken")
    sources = [f["source"] for f in out]

    assert sources.index("pkg/util.py") < sources.index("tests/test_auth.py")
    assert sources.index("docs/auth.rst") > sources.index("tests/test_auth.py")
    assert len(out) == len(fragments)
    assert out[-1]["content"] == "unkeyed"


def test_optimize_context_applies_engine_s6_postpass(monkeypatch):
    calls = []

    def fake_localize_fragments(fragments, query, *, k=None):
        calls.append((query, len(fragments), k))
        return list(reversed(fragments))

    monkeypatch.setattr(
        "entroly.file_localizer.localize_fragments",
        fake_localize_fragments,
    )

    engine = EntrolyEngine()
    engine.ingest_fragment("def a(): return 1", "a.py", 8)
    engine.ingest_fragment("def b(): return 2", "b.py", 8)

    result = engine.optimize_context(token_budget=1000, query="fix a")

    assert calls, "optimize_context did not invoke engine_s6 post-pass"
    selected = result.get("selected_fragments") or result.get("selected") or []
    assert len(selected) >= 2
    assert result.get("selected") == result.get("selected_fragments")
