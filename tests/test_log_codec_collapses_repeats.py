"""Log compression must collapse repeats and keep the first causal error.

Deduplication keyed on the raw line cannot collapse repeats that carry a
counter -- and that is exactly the shape that matters: one root-cause error
followed by hundreds of downstream retries, each with a different retry number.

Measured on the fixture below (203 lines, 16,311 chars), 200 lines reading
"request failed: connection pool exhausted (retry N)" were 200 distinct keys,
so none collapsed. The output stayed long enough that a downstream summariser
ran and concatenated log entries together, destroying line structure as well.

Templating variable runs before keying (Drain, He et al. ICWS 2017; LogPai)
took the same input to 308 chars with line structure intact. The first
occurrence is emitted verbatim, so the representative line keeps its real
values -- only the KEY is lossy.
"""

from __future__ import annotations

import pytest

from entroly.universal_compress import detect_content_type, universal_compress

ROOT_CAUSE = (
    "2026-08-02T10:00:03Z ERROR db connect failed: FATAL password "
    "authentication failed for user 'svc_billing'"
)


def _log_text(repeats: int = 200) -> str:
    lines = ["2026-08-02T10:00:00Z INFO  worker starting pool_size=8", ROOT_CAUSE]
    for i in range(repeats):
        lines.append(
            f"2026-08-02T10:00:{4 + i % 50:02d}Z ERROR request failed: "
            f"connection pool exhausted (retry {i})"
        )
    lines.append("2026-08-02T10:05:00Z INFO  worker shutting down exit_code=70")
    return "\n".join(lines)


def _compress(text: str) -> str:
    out = universal_compress(text, target_ratio=0.3)
    return out[0] if isinstance(out, tuple) else str(out)


def test_detects_log():
    assert detect_content_type(_log_text()) == "log"


def test_first_causal_error_survives_the_flood():
    out = _compress(_log_text())
    assert "password authentication failed" in out, (
        "the root cause was lost among its own downstream repeats -- the one "
        f"line that explains the incident.\n---\n{out[:500]}"
    )
    assert "svc_billing" in out, "the representative line must keep exact values"


def test_repeats_collapse_to_one_line_with_a_count():
    out = _compress(_log_text(repeats=200))
    assert "[×200]" in out or "[x200]" in out, (
        f"200 instances of one event should collapse to a counted line.\n"
        f"---\n{out[:500]}"
    )
    assert out.count("connection pool exhausted") == 1, (
        "the repeated event should appear exactly once"
    )


def test_line_structure_is_preserved():
    """Log entries must stay on their own lines to remain parseable."""
    out = _compress(_log_text())
    for line in out.splitlines():
        assert line.count("2026-08-02T") <= 1, (
            f"multiple log entries were concatenated onto one line, which "
            f"destroys entry boundaries:\n  {line[:200]}"
        )


def test_boundaries_are_kept():
    out = _compress(_log_text())
    assert "pool_size=8" in out, "the start-of-run line is a boundary"
    assert "exit_code=70" in out, "the exit status is a boundary and an outcome"


@pytest.mark.parametrize(
    "varying",
    [
        "request 12345 failed",
        "request 99999 failed",
        "conn 0x7ffe1234 dropped",
        "conn 0x7ffe9999 dropped",
        "peer 10.0.0.4:8080 reset",
        "peer 10.0.0.9:9090 reset",
        "took 12.5ms",
        "took 99.1ms",
        "job 3f2a1b4c-1111-2222-3333-444455556666 done",
    ],
)
def test_template_normalises_variable_runs(varying):
    """Counters, hex, addresses, durations and uuids must not defeat collapse."""
    from entroly.universal_compress import _log_template

    template = _log_template(varying)
    assert "*" in template, f"{varying!r} produced no placeholder: {template!r}"


def test_distinct_events_do_not_collapse_into_each_other():
    """Templating must not merge genuinely different messages."""
    text = "\n".join(
        [
            "2026-08-02T10:00:00Z ERROR disk full on /var/log",
            "2026-08-02T10:00:01Z ERROR permission denied on /etc/shadow",
        ]
    )
    out = _compress(text)
    assert "disk full" in out and "permission denied" in out, (
        f"two different errors were collapsed into one.\n---\n{out}"
    )
