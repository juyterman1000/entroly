# Defect and evidence register

| ID | Classification | Severity | Status | Proof / mitigation |
| --- | --- | --- | --- | --- |
| DOGFOOD-001 | Malformed pricing catalog (NaN, infinity, invalid numeric) can escape fail-closed validation | Medium | FIX IN PR; CI PENDING | `net_savings._rate` failed to catch `decimal.InvalidOperation` and accepted positive infinity; fix rejects nonfinite rates; `tests/test_net_savings.py` adds regression coverage |
| GAP-001 | No independent live provider vs baseline agent quality/cost comparison in this audit | Evidence gap | NOT TESTED | `WORKLOAD_MATRIX.md` |
| GAP-002 | No verified real invoice reconciliation or output-cost accounting | Evidence gap | NOT TESTED | `COST_AND_QUALITY.md` |
| GAP-003 | Full repo-wide attack and dependency-graph analysis not performed here | Evidence gap | NOT TESTED | `TRUST_AUDIT.md` |

No other newly reproduced defects are claimed in this investigation.

## DOGFOOD-001 reproduction

The old code executes `value = Decimal(str(rate))` followed by `if value < 0`. For `NaN` and invalid strings, `decimal.InvalidOperation` escapes the catalog loader; `Infinity` is accepted. See `REPRODUCE.md` for the minimal proof and tests.

## Priority rationale

Receipt truth and controlled fail-closed behavior outrank cosmetic dashboard improvements. The supplied catalog file is an operator-controlled input, so this is not described as remotely exploitable without evidence.
