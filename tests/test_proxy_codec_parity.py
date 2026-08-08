"""The proxy must preserve the evidence the codecs protect.

`proxy_transform` carried no codec import, so tool results -- JSON payloads,
logs, shell and test output, tables -- were compressed by keyword-pattern rules
instead of the registry built for exactly those shapes. Measured on the
`codec_ablation` fixtures, that path reduced 75.1% while retaining **24.0%** of
required evidence, against 76.9% / 100.0% for the registry: it reached its ratio
by deleting the failures and identifiers the codecs exist to protect, and was
worse than blind truncation.

These tests pin the repaired behaviour. They are written against the same
fixtures and required-evidence lists the codec work used, so a regression here
means the highest-traffic surface has started dropping evidence again.
"""

from __future__ import annotations

import pytest

from benchmarks.codec_ablation import FIXTURES
from entroly.proxy_transform import _compress_build_errors, compress_tool_output


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_proxy_preserves_required_evidence(name: str) -> None:
    """Every load-bearing value survives proxy compression of a tool result."""
    text, required = FIXTURES[name]()
    compressed, kind, _savings = compress_tool_output(text)

    missing = [item for item in required if item not in compressed]
    assert not missing, (
        f"{name}: proxy compression ({kind}) dropped {len(missing)} of "
        f"{len(required)} required items: {missing[:3]}"
    )


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_proxy_actually_compresses(name: str) -> None:
    """Preservation must not be bought by declining to compress at all.

    The failure mode the codec work started from was a codec that could only
    destroy values or refuse, and defaulted to refusing -- invisible, because
    0% compression raises no alarm.
    """
    text, _required = FIXTURES[name]()
    compressed, _kind, savings = compress_tool_output(text)

    assert len(compressed) < len(text), f"{name}: no compression applied"
    assert savings > 0.10, f"{name}: savings {savings:.1%} below the 10% floor"


def test_codec_fault_falls_back_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """A codec fault must not turn a proxy call into an error."""
    import entroly.codecs_builtin as builtin

    def boom(*_args, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(builtin, "default_registry", boom)

    text, _required = FIXTURES["log_root_cause_flood"]()
    compressed, kind, _savings = compress_tool_output(text)

    assert isinstance(compressed, str) and compressed
    assert kind != "codec"


def test_rollback_flag_restores_pattern_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ENTROLY_PROXY_CODECS=0` is the documented rollback for live traffic."""
    text, _required = FIXTURES["log_root_cause_flood"]()

    monkeypatch.setenv("ENTROLY_PROXY_CODECS", "0")
    disabled, disabled_kind, _ = compress_tool_output(text)

    monkeypatch.setenv("ENTROLY_PROXY_CODECS", "1")
    enabled, enabled_kind, _ = compress_tool_output(text)

    assert disabled_kind != "codec"
    assert enabled_kind == "codec"
    assert disabled != enabled


def test_short_results_are_left_alone() -> None:
    """Below the minimum length the contract is an untouched passthrough."""
    tiny = "ok"
    assert compress_tool_output(tiny) == (tiny, "none", 0.0)


# ── build-diagnostic detection must not fire on a tool name in a path ──────


def test_file_listing_is_not_treated_as_build_output() -> None:
    """A path containing a linter's name must not rewrite an unrelated result.

    Detection was `any(kw in content ...)` over the whole blob, including the
    bare tool names "ruff", "tsc" and "eslint". A real `git ls-files` listing
    of 1,316 lines -- containing no errors -- matched on the substring "ruff"
    in a path, kept only the two filenames containing "error", and emitted
    "[entroly: 2 errors, 0 warnings - 1315 lines compressed]". That is 99.7% of
    the evidence destroyed and a fabricated error count attached to content
    that had none.
    """
    listing = "\n".join(
        [
            "pyproject.toml",
            "ruff.toml",  # the tool name that used to trigger detection
            "src/tsconfig.json",
            "benchmarks/fever_error_analysis.py",  # 'error' inside a filename
            "tests/test_snapshot_errors.py",
        ]
        + [f"src/module_{i}.py" for i in range(300)]
    )

    assert _compress_build_errors(listing) is None

    compressed, kind, _savings = compress_tool_output(listing)
    assert kind != "build_errors"
    assert "0 warnings" not in compressed, "fabricated a diagnostic summary"


@pytest.mark.parametrize(
    "sample",
    [
        "error[E0499]: cannot borrow `x` as mutable more than once",
        "src/lib.rs:42:9: error: mismatched types",
        "warning: unused variable: `y`",
        "SyntaxError: invalid syntax",
    ],
)
def test_real_diagnostics_are_still_detected(sample: str) -> None:
    """Tightening detection must not stop it recognising actual build output."""
    noise = "\n".join(f"   Compiling dep{i} v0.1.0" for i in range(120))
    assert _compress_build_errors(f"{sample}\n{noise}") is not None


def test_prose_is_not_build_output() -> None:
    assert _compress_build_errors("hello world\n" * 80) is None
