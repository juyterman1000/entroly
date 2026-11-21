"""The built-in proof-of-value queries must actually discriminate.

`entroly simulate` and `entroly demo` are what the one-time welcome banner tells
a new user to run to decide whether to adopt. Their default queries were phrased
as natural-language questions -- "How does the authentication flow work?" --
and question words plus stopwords dilute the lexical score until nothing
discriminates. Selection then fills the budget with everything, which scores
~0% against a fixed baseline.

Measured on a 4-file fixture repository, the three question-phrased defaults
each returned byte-identical results and headlined **1.3%**, while keyword
phrasing over the identical corpus headlined **69.0%**. The engine was never the
problem; the defaults were understating it by roughly 58x at the one moment a
new user is deciding.

These tests pin the two properties that keep that from coming back: the
defaults are keyword-shaped, and they select different, topically correct
context. They assert *discrimination*, never a particular percentage, so
ranker tuning cannot make them flaky.
"""

from __future__ import annotations

import json

import pytest

from entroly.cli import _DEFAULT_SIMULATION_QUERIES, _simulation_queries
from entroly.cli_parser import GETTING_STARTED

#: Leading words that make a string a question rather than a set of search
#: terms. These are exactly what diluted the old defaults.
_INTERROGATIVES = {"how", "what", "why", "where", "when", "which", "who",
                   "explain", "describe", "find", "tell", "show", "does", "do",
                   "is", "are", "can", "should"}

#: One document per default query, each unambiguously about one topic.
_CORPUS = {
    "auth_session.py": (
        "def authenticate(login, token):\n"
        "    session = create_session(login, token)\n"
        "    return session.token\n"
        "def refresh_session(session, token):\n"
        "    return session.renew(token)\n"
    ),
    "error_paths.py": (
        "def validate(amount):\n"
        "    if amount <= 0:\n"
        "        raise ValueError('validation failed')\n"
        "    try:\n"
        "        commit(amount)\n"
        "    except RuntimeError as exception:\n"
        "        raise exception\n"
    ),
    "package_layout.py": (
        "import collections\n"
        "import importlib\n"
        "from mypkg.module import dependencies\n"
        "PACKAGE_STRUCTURE = {'mypkg': ['module', 'dependencies']}\n"
    ),
}

#: Which document each default query must prefer, in declaration order.
_EXPECTED_TOPIC = ["auth_session.py", "error_paths.py", "package_layout.py"]


def test_the_defaults_are_search_terms_not_questions() -> None:
    """A question-shaped default is the defect; reject the shape outright."""
    for query in _DEFAULT_SIMULATION_QUERIES:
        assert "?" not in query, f"{query!r} is phrased as a question"
        first = query.split()[0].lower()
        assert first not in _INTERROGATIVES, (
            f"{query!r} opens with the interrogative {first!r}; "
            "question words dilute the lexical score until nothing ranks"
        )


def test_an_explicit_query_still_wins() -> None:
    """--query must override the defaults; only the no-query path changed."""
    assert _simulation_queries(_Args(query=["my own task"])) == ["my own task"]
    assert _simulation_queries(_Args(query=[])) == list(_DEFAULT_SIMULATION_QUERIES)
    assert _simulation_queries(_Args(query="  ")) == list(_DEFAULT_SIMULATION_QUERIES)


class _Args:
    def __init__(self, query) -> None:
        self.query = query


@pytest.mark.parametrize(
    "query,expected",
    list(zip(_DEFAULT_SIMULATION_QUERIES, _EXPECTED_TOPIC)),
    ids=_EXPECTED_TOPIC,
)
def test_each_default_selects_its_own_topic(query, expected, tmp_path) -> None:
    """Each default must rank its matching document first.

    This is the property that broke: the old defaults ranked nothing, so every
    query returned the same full set. A fresh store per run matters -- reusing
    the repository's own index would score against the wrong corpus.
    """
    from entroly.sdk import create_context_receipt

    receipt = create_context_receipt(
        _CORPUS,
        query=query,
        budget=60,
        chunk_tokens=40,
        overlap_tokens=0,
        store_dir=str(tmp_path),
    )
    selected = receipt.get("selected_context") or []
    assert selected, f"{query!r} selected nothing from the fixture corpus"
    top = str(selected[0].get("source_path", ""))
    assert top == expected, (
        f"{query!r} ranked {top!r} first, expected {expected!r}; "
        "the default no longer discriminates"
    )


def test_the_defaults_do_not_all_select_the_same_context() -> None:
    """The exact signature of the defect: identical results for every query."""
    from entroly.sdk import create_context_receipt

    signatures = set()
    for query in _DEFAULT_SIMULATION_QUERIES:
        receipt = create_context_receipt(
            _CORPUS, query=query, budget=60, chunk_tokens=40, overlap_tokens=0,
        )
        selected = receipt.get("selected_context") or []
        signatures.add(json.dumps(
            sorted(str(chunk.get("source_path", "")) for chunk in selected)
        ))
    assert len(signatures) > 1, (
        "every default selected the same context, which is what headlined "
        "1.3% on a repository where keyword queries headlined 69.0%"
    )


def test_help_keeps_the_getting_started_path() -> None:
    """The welcome banner fires once, to stderr. --help is the only way back.

    The banner signs off with "Run entroly --help anytime" while help carried
    none of the path it points at: a user who pipes stdout or scrolls past lost
    onboarding permanently.
    """
    for command in ("entroly verify-claims", "entroly simulate",
                    "claude mcp add entroly", "entroly doctor"):
        assert command in GETTING_STARTED, f"--help lost {command!r}"
    assert "--query" in GETTING_STARTED, (
        "help must surface --query; the sample-query caveat was dim grey text "
        "that the headline percentage contradicted"
    )


def test_share_measures_with_the_same_queries_as_simulate() -> None:
    """A published figure must not be measured with weaker queries.

    `cmd_share` carried its own copy of the question-phrased list, so fixing
    `simulate` alone would have left the shareable number understated.
    """
    import inspect

    from entroly import cli

    source = inspect.getsource(cli.cmd_share)
    assert "_DEFAULT_SIMULATION_QUERIES" in source, (
        "cmd_share must reuse the shared defaults, not redeclare them"
    )
