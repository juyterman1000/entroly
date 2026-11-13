# Adversarial trust audit — bounded findings

Scope: `entroly/net_savings.py`, bundled pricing tables, selected proxy fidelity tests and CI workflow definitions. This is **not** a comprehensive exploit assessment of the entire codebase.

## Confirmed parser defect (addressed in this PR)

At baseline, `_rate(raw, key)` called `Decimal(str(raw[key]))` and then `value < 0`, without handling `decimal.InvalidOperation` or checking `Decimal.is_finite()`.

- `Decimal("NaN") < 0` raises `decimal.InvalidOperation`, which is not in the caught catalog loader exception types.
- `Decimal("not-a-number")` also raises `decimal.InvalidOperation`.
- `Decimal("Infinity")` passes the old nonnegative validation and could enter subsequent money calculations.

This is a **fail-closed and availability problem**, especially with `ENTROLY_PRICING_CATALOG` configured. No evidence indicates it is remotely injectable by an ordinary API user; a local operator-supplied or damaged rate table is the trigger.

The patch converts bad or nonfinite rates to `ValueError`. The existing catalog loaders already catch `ValueError` and decline to price the invalid table. Tests cover operator and bundled per-provider isolation.

## Existing safeguards inspected

- Pricing is source-linked and model-specific; unsupported models remain unpriced.
- `PromptInventory` retains digests and byte counts, not raw prompt strings.
- Proxy tests cover file reads and unexpected context injection.
- Repo policy requires exact recovery, WITNESS/RAVS fail-closed, local-first, and stable cache prefixes.

These statements describe source inspection and CI coverage, not an independent adversarial certification.

## Unexecuted trust gates

| Threat | State |
| --- | --- |
| Concurrent live requests with mixed providers/credentials | NOT EXECUTED |
| Actual secret exfiltration/redaction across all entrypoints | NOT EXECUTED |
| Malicious text in tool outputs inducing policy violations | NOT EXECUTED |
| Broken store and mid-request crash with exact recovery | NOT EXECUTED |
| Cross-agent digest/cache attribution under concurrent loads | NOT EXECUTED |
| Native engine missing/incompatible in installed packages | CI COVERAGE ONLY |
| Full WITNESS/RAVS proof replay and independently checked receipts | CI COVERAGE ONLY |
| Actual billing rollback and pricing tier reconciliation | NOT EXECUTED |

These remaining items must stay visible, not be silently promoted to PASS.
