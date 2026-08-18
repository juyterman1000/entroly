"""What `entroly simulate` reports when the native engine is missing.

Query-conditioned selection (QCCR) is gated on the native engine: its candidate
set comes from `self._rust.export_fragments()`, which has no pure-Python
equivalent, so `optimize_context` skips relevance ranking entirely without it.
The optimizer still returns fragments and still fills the token budget -- it just
never reads the query.

The reported percentage is `(baseline - selected_tokens) / baseline` with
`baseline = min(total_tokens, 32_000)`, so it is decided by the budget before
selection happens. Measured on this repo with the engine absent, three unrelated
questions -- including the nonsense control "banana bicycle weather forecast tuna
sandwich" -- produced a byte-identical 23 fragments / 7,588 tokens and the same
"76.29% saved".

Normally nobody sees this: `_auto_repair_before_measuring` installs the engine
first. These tests cover the paths where repair cannot run -- offline, CI,
PEP 668, `ENTROLY_NO_SELF_HEAL=1`.

The contract is **label, do not withhold**. An earlier version suppressed the
number entirely. That was honest and useless: a first run that reports nothing
gives the user no reason to continue. So the figure is printed, and the caveat
is printed in the same block, because a claim and its caveat have to travel
together or the caveat does not exist.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from entroly import self_heal
from entroly.cli import _print_local_simulation


def _report(*, query_conditioned: bool) -> dict:
    limitations = ["No LLM call was made; quality is not judged here."]
    if not query_conditioned:
        limitations.append(
            "The native engine is unavailable, so selection did not read the "
            "query: these figures are budget arithmetic and are NOT a measured "
            'saving. Install it with: pip install -U "entroly[native]"'
        )
    return {
        "files_indexed": 140,
        "repo_tokens_indexed": 604_158,
        "baseline_tokens_per_query": 32_000,
        "budget": 8_000,
        "budget_narrowed_to_demonstrate": False,
        "average_reduction_pct": 76.29,
        "total_tokens_saved": 24_412,
        "query_conditioned_selection": query_conditioned,
        "selection_engine": "qccr-native" if query_conditioned else "python-fallback-unranked",
        "queries": [
            {
                "query": "why did this project stop growing",
                "selected_fragments": 23,
                "selected_tokens": 7_588,
                "reduction_pct": 76.29,
                "tokens_saved": 24_412,
                "latency_ms": 15.96,
                "top_sources": ["file:pkg/rare.py", "file:src/qccr.rs"],
            }
        ],
        "latency_ms": {"min": 15.96, "p95": 15.96, "max": 15.96},
        "limitations": limitations,
    }


def _render(report: dict) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_local_simulation(report, title="Entroly Simulate", include_perf=False)
    return buf.getvalue()


# ── degraded output ──────────────────────────────────────────────────────────

def test_degraded_run_still_shows_the_number() -> None:
    """Withholding the figure leaves a first-time user with nothing."""
    out = _render(_report(query_conditioned=False))

    assert "76.3%" in out
    assert "24,412" in out


def test_degraded_number_is_labelled_unearned_in_the_same_block() -> None:
    """The caveat must not be somewhere else on the page.

    Printing "76.3%" and burying the qualification in a trailing limitations
    line is how an unearned number ends up quoted on its own.
    """
    out = _render(_report(query_conditioned=False))

    assert "unearned" in out
    headline = next(
        line for line in out.splitlines() if "Average reduction" in line
    )
    assert "unearned" in headline, (
        f"the caveat must be on the headline itself, got: {headline!r}"
    )
    assert "not a measured saving" in headline


def test_degraded_run_says_why_and_how_to_fix() -> None:
    out = _render(_report(query_conditioned=False))

    assert "Relevance ranking is OFF" in out
    assert "did not read your query" in out
    assert 'pip install -U "entroly[native]"' in out


def test_healthy_run_carries_no_caveat() -> None:
    """The warning must not leak into a correctly-installed run."""
    out = _render(_report(query_conditioned=True))

    assert "Average reduction: 76.3%" in out
    assert "24,412 tokens saved" in out
    assert "Relevance ranking is OFF" not in out
    assert "unearned" not in out
    assert "entroly[native]" not in out


def test_report_exposes_selection_mode_for_machine_consumers() -> None:
    """`--json` consumers must be able to tell the two modes apart."""
    degraded = _report(query_conditioned=False)
    healthy = _report(query_conditioned=True)

    assert degraded["query_conditioned_selection"] is False
    assert degraded["selection_engine"] == "python-fallback-unranked"
    assert any("NOT a measured saving" in line for line in degraded["limitations"])

    assert healthy["query_conditioned_selection"] is True
    assert healthy["selection_engine"] == "qccr-native"
    assert not any("NOT a measured saving" in line for line in healthy["limitations"])


# ── self-heal gating ─────────────────────────────────────────────────────────

def test_repair_is_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Locked, audited, and air-gapped environments need a hard off switch."""
    monkeypatch.setenv(self_heal.ENV_DISABLE, "1")
    monkeypatch.setattr(self_heal, "native_engine_ready", lambda: False)

    outcome = self_heal.repair_native()

    assert outcome.attempted is False
    assert outcome.blocked is True
    assert self_heal.ENV_DISABLE in outcome.blocked_reason


def test_repair_does_not_recurse_after_reexec(monkeypatch: pytest.MonkeyPatch) -> None:
    """The re-executed process must not try to repair again.

    Without this guard a repair that installs successfully but still fails the
    readiness check would re-exec forever.
    """
    monkeypatch.setenv(self_heal.ENV_GUARD, "1")
    monkeypatch.setattr(self_heal, "native_engine_ready", lambda: False)

    outcome = self_heal.repair_native()

    assert outcome.attempted is False
    assert outcome.blocked is True


def test_repair_is_a_noop_when_the_engine_is_already_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(self_heal, "native_engine_ready", lambda: True)

    outcome = self_heal.repair_native()

    assert outcome.healed is True
    assert outcome.attempted is False
    assert outcome.needs_reexec is False


def test_externally_managed_python_is_never_written_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PEP 668 refusal must degrade to a labelled report, not an exception.

    Debian/Ubuntu and Homebrew system Pythons ship an EXTERNALLY-MANAGED marker.
    Installing there needs --break-system-packages, which Entroly must never
    pass on a user's behalf.
    """
    monkeypatch.setattr(self_heal, "_externally_managed", lambda: True)

    assert self_heal._installer_command() is None

    ok, detail = self_heal.install_native_engine()
    assert ok is False
    assert "externally managed" in detail


@pytest.mark.parametrize(
    "call",
    [
        pytest.param("server", id="mcp-server"),
        pytest.param("proxy", id="proxy"),
    ],
)
def test_services_warn_when_repair_is_switched_off(
    call: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Disabling repair must not make a degraded service silent.

    Caught by hand: the first version short-circuited on `disabled()` before
    printing anything, so an operator who set ENTROLY_NO_SELF_HEAL got a server
    that returned query-independent context indefinitely with no indication why.
    A one-shot CLI run has the labelled report to fall back on; a long-lived
    service has nothing.
    """
    monkeypatch.setenv(self_heal.ENV_DISABLE, "1")
    monkeypatch.setattr(self_heal, "native_engine_ready", lambda: False)

    if call == "server":
        from entroly.server import _repair_native_engine_at_startup

        _repair_native_engine_at_startup()
    else:
        from entroly.cli import _auto_repair_for_service

        assert _auto_repair_for_service("proxy") is None

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "will not read the query" in err
    assert self_heal.ENV_DISABLE in err


def test_service_repair_never_writes_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The MCP server speaks JSON-RPC on stdout.

    One stray progress line there desynchronises the client, so every repair
    message on a service path must go to stderr.
    """
    monkeypatch.setenv(self_heal.ENV_DISABLE, "1")
    monkeypatch.setattr(self_heal, "native_engine_ready", lambda: False)

    from entroly.server import _repair_native_engine_at_startup

    _repair_native_engine_at_startup()

    captured = capsys.readouterr()
    assert captured.out == "", (
        f"MCP stdout must carry protocol only, got: {captured.out!r}"
    )
    assert captured.err


def test_importing_entroly_does_not_install_anything() -> None:
    """An import must never shell out to pip.

    Installing mid-import risks re-entrant imports, partially initialised state,
    and multiprocessing deadlock, so the SDK exposes `entroly.repair()` instead
    of healing implicitly.
    """
    import entroly

    assert hasattr(entroly, "repair")
    assert hasattr(entroly, "native_engine_ready")
    assert self_heal._attempted_in_this_process is False
