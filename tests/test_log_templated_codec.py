"""Templated logs must compress without merging distinct values.

The log codec collapsed *identical* lines and nothing else. ``_log_template``
masks only the final numeric run, so two lines differing anywhere earlier never
share a template. Measured on 400 lines:

    400 identical lines   -> 100% reduction
    400 templated lines   ->   0% reduction

Real logs are templated, not duplicated, so the codec did nothing on the
common case. The conservatism was itself a fix: an earlier version collapsed
every digit run and merged ``code 402`` with ``code 500``, which is a far worse
failure than not compressing.

The templated form takes neither side. It masks every varying slot so lines
group, and keeps the values verbatim in per-slot columns rather than discarding
them, so distinct codes stay distinct. These tests fix that property.
"""

from __future__ import annotations

import re

import pytest

from entroly.codec import RecoveryStore
from entroly.codecs_builtin import default_registry


def _templated(text: str, store: RecoveryStore | None = None):
    # `store or RecoveryStore()` would be wrong: RecoveryStore defines
    # __len__, so an empty one is falsy and the caller's store would be
    # silently replaced, leaving recovery handles pointing at a discarded
    # instance. default_registry() avoids the same trap the same way.
    store = store if store is not None else RecoveryStore()
    reps = default_registry(store).representations(
        text, source_id="log", content_type="", query=""
    )
    return next(
        (r for r in reps if r.representation_id.endswith("#log.templated")), None
    )


UNIFORM = "\n".join(
    f"2026-08-06T10:{i % 60:02d}:00 INFO worker={i % 8} processed batch {i} in {i % 40}ms"
    for i in range(400)
)

WITH_ERRORS = "\n".join(
    (
        f"2026-08-06T10:{i % 60:02d}:00 ERROR db timeout code "
        f"{402 if i % 100 == 0 else 500} on shard {i % 4}"
        if i % 50 == 0
        else f"2026-08-06T10:{i % 60:02d}:00 INFO ok {i}"
    )
    for i in range(400)
)


def test_templated_logs_actually_compress() -> None:
    """The measured defect: this shape used to reduce by 0%."""
    rep = _templated(UNIFORM)
    assert rep is not None, "no templated representation offered"
    assert rep.token_cost < (len(UNIFORM) // 4) * 0.8


def test_distinct_status_codes_do_not_merge() -> None:
    """The failure this codec was made conservative to avoid."""
    rep = _templated(WITH_ERRORS)
    assert rep is not None
    assert "402" in rep.text
    assert "500" in rep.text


def test_error_lines_are_never_templated() -> None:
    """An error is why someone is reading the log; it stays as written."""
    rep = _templated(WITH_ERRORS)
    assert rep is not None
    source_errors = [ln for ln in WITH_ERRORS.splitlines() if "ERROR db timeout" in ln]
    for line in source_errors:
        assert line in rep.text, f"error line was altered or dropped: {line!r}"


def test_every_varying_value_survives() -> None:
    """Templating must factor the invariant part only."""
    rep = _templated(WITH_ERRORS)
    assert rep is not None
    counters = set(re.findall(r"\bok (\d+)", WITH_ERRORS))
    missing = [c for c in counters if c not in rep.text]
    assert not missing, f"{len(missing)} counter values were lost, e.g. {missing[:5]}"


def test_the_original_is_recoverable() -> None:
    store = RecoveryStore()
    rep = _templated(UNIFORM, store)
    assert rep is not None
    assert rep.recovery is not None
    assert store.recover(rep.recovery) == UNIFORM


def test_short_logs_are_left_alone() -> None:
    """Below the group threshold the template costs more than it saves."""
    assert _templated("a 1\nb 2\n") is None


def test_lines_without_varying_values_are_kept_verbatim() -> None:
    """Nothing to factor; the existing duplicate collapse handles these."""
    text = "\n".join("service ready" for _ in range(50))
    rep = _templated(text)
    if rep is not None:
        assert "service ready" in rep.text


def test_declines_when_a_value_would_be_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    """The multiset check must gate the representation, not decorate it."""
    import entroly.codecs_builtin as builtin

    real = builtin._template_factored_log

    def lossy(text: str, is_critical):  # noqa: ANN001
        rendered = real(text, is_critical)
        if rendered is None:
            return None
        # Drop one value from the first emitted column.
        out = []
        dropped = False
        for line in rendered.splitlines():
            if not dropped and line.strip().startswith("{0}:"):
                head, _, values = line.partition(":")
                out.append(f"{head}:{','.join(values.strip().split(',')[:-1])}")
                dropped = True
            else:
                out.append(line)
        return "\n".join(out)

    monkeypatch.setattr(builtin, "_template_factored_log", lossy)
    assert _templated(UNIFORM) is None, (
        "a rendering that dropped a value was accepted; the preservation "
        "check does not actually gate the representation"
    )
