"""CSV / TSV codec: keep the schema and the shape, drop the bulk.

A table is read for two different things and a compressor must serve both:

* **the contract** -- what columns exist, what type each holds, which are
  identifiers. Losing a column name makes every remaining row unusable.
* **the shape** -- how large, how sparse, what range, what the extremes are.
  A caller asking "did anything go wrong in this export" needs the outliers
  and the missingness, not row 4,000.

So this keeps the header verbatim, infers a type per column, keeps the first
and last rows as representatives, and summarises each numeric column by count,
missing, min, median, max. Everything dropped goes to the recovery store, so
the rows are not gone -- they are elsewhere, addressable, and verifiable.

Deliberately NOT done: group-by summaries, change-point detection, referential
integrity across files, and spreadsheet input. Those need either a schema the
caller supplies or a second file to join against, and guessing at them would
produce confident nonsense.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from .codec import (
    RecoveryStore,
    Representation,
    SupportDecision,
    content_digest,
    estimate_tokens,
)

# Rows kept verbatim at each end. Two is enough to show the shape of a record
# without implying the reader has seen a sample.
_EDGE_ROWS = 2
_MIN_ROWS = 6


def _sniff(text: str) -> tuple[str, list[list[str]]] | None:
    """Return (delimiter, rows) when the text parses as a consistent table."""
    sample = text[:8192]
    counts = {d: sample.count(d) for d in (",", "\t", ";", "|")}
    delimiter = max(counts, key=lambda d: counts[d])
    if counts[delimiter] == 0:
        return None
    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except (csv.Error, ValueError):
        return None
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if len(rows) < _MIN_ROWS:
        return None
    width = len(rows[0])
    if width < 2:
        return None
    # A table has a stable width. Prose with commas does not.
    consistent = sum(1 for r in rows if len(r) == width)
    if consistent / len(rows) < 0.9:
        return None
    return delimiter, rows


def _as_number(cell: str) -> float | None:
    try:
        return float(cell.replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _column_type(values: list[str]) -> str:
    present = [v for v in values if v.strip()]
    if not present:
        return "empty"
    if all(_as_number(v) is not None for v in present):
        return "number"
    if all(v.strip().lower() in {"true", "false", "yes", "no", "0", "1"} for v in present):
        return "boolean"
    if len(set(present)) == len(present):
        return "unique"      # identifier-like: every value distinct
    return "text"


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _summarise(header: list[str], body: list[list[str]]) -> list[str]:
    out = []
    for i, name in enumerate(header):
        values = [r[i] if i < len(r) else "" for r in body]
        missing = sum(1 for v in values if not v.strip())
        kind = _column_type(values)
        line = f"  {name}: {kind}, {len(values)} rows"
        if missing:
            line += f", {missing} missing"
        if kind == "number":
            nums = sorted(n for n in (_as_number(v) for v in values) if n is not None)
            if nums:
                line += (
                    f", min={nums[0]:g}, p50={_quantile(nums, 0.5):g}, max={nums[-1]:g}"
                )
        elif kind in {"text", "boolean"}:
            distinct = len({v for v in values if v.strip()})
            line += f", {distinct} distinct"
        out.append(line)
    return out


class TableCodec:
    """CSV/TSV: header verbatim, per-column summary, edge rows, exact recovery."""

    name = "table"
    version = "1"

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type in {"csv", "tsv", "table"}:
            return SupportDecision(True, 1.0, "declared content type")
        if text.lstrip().startswith(("{", "[")):
            return SupportDecision(False, 0.0, "looks like JSON")
        sniffed = _sniff(text)
        if sniffed is None:
            return SupportDecision(False, 0.0, "no consistent delimited table found")
        return SupportDecision(True, 0.85, f"{len(sniffed[1])} consistent rows")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        src_digest = content_digest(text)
        reps = [
            Representation(
                representation_id=f"{source_id}#table.full",
                source_id=source_id,
                content_type="table",
                text=text,
                token_cost=estimate_tokens(text),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                distortion_risk=0.0,
            )
        ]

        sniffed = _sniff(text)
        if sniffed is None:
            return reps
        delimiter, rows = sniffed
        header, body = rows[0], rows[1:]
        if len(body) <= _EDGE_ROWS * 2:
            return reps

        lines = text.split("\n")
        joined = delimiter.join
        kept_lines = [joined(header)]
        kept_lines += [joined(r) for r in body[:_EDGE_ROWS]]
        kept_lines += [f"... {len(body) - _EDGE_ROWS * 2} rows elided ..."]
        kept_lines += [joined(r) for r in body[-_EDGE_ROWS:]]
        kept_lines += ["", f"# {len(body)} data rows, {len(header)} columns"]
        kept_lines += _summarise(header, body)

        summary = "\n".join(kept_lines)
        if len(summary) >= len(text):
            return reps

        # Recovery stores the WHOLE original, so recover() returns the exact
        # byte stream rather than a reconstruction that has to be trusted.
        recovery = self.store.put(
            text,
            item_count=len(body),
            note=f"full table for {source_id or 'csv'}",
        )

        protected = tuple(h for h in header if h and h in summary)
        reps.append(
            Representation(
                representation_id=f"{source_id}#table.summary",
                source_id=source_id,
                content_type="table",
                text=summary,
                token_cost=estimate_tokens(summary),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                protected_evidence=protected,
                distortion_risk=1.0 - (len(summary) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        del lines
        return reps
