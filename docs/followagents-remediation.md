# FollowAgents remediation record

Status: changes under review; no patch release or external reassessment claimed.
The correction request below is a draft and has not been sent.

## Reviewed evidence

The [public listing](https://followagents.com/en/agents/entroly), checked on
2026-09-14, shows 68/100, “Use with care”, low evidence confidence, and reviewed
revision `f99a6c80ed9618e440fbb1d126a540403f3bf784` dated 2026-09-08.
This patch starts from public main `e6fdc89eacd8d3bbd962d845001fa2e5496666af`.

| Concern | Classification | Remediation / remaining boundary |
| --- | --- | --- |
| Unconfirmed native package install | Runtime consent defect | Default repair disabled; explicit SDK/environment consent; hard-off overrides; PEP 668 path fixed. |
| Recoverable source stored locally | Security/documentation | Atomic receipt writes, private POSIX files, documented ACL, retention, scope, and deletion limits. No encryption claim. |
| Unsupported public assertions | Claim quality | Removed absolute accuracy/uniqueness claims; disclosed weak historical evidence, observed losses, telemetry and provider egress. |
| Changelog absent from review evidence | Evidence coverage plus stale links | File existed at reviewed revision; current links repaired and unreleased changes separated from published releases. |
| Single maintainer identity | Organizational limitation | Not solved by this patch. No invented identity, affiliation, or maintenance guarantee. |
| Independent effectiveness validation | External evidence limitation | Not solved by unit tests. Independent task-quality replication remains necessary. |

The [changelog at the reviewed revision](https://github.com/juyterman1000/entroly/blob/f99a6c80ed9618e440fbb1d126a540403f3bf784/CHANGELOG.md)
supports a narrow evidence correction. It does not establish that all historical
links worked or that the reviewer had inspected the file. Current-main cumulative
download and disputed model-name claims had already been removed before this patch.

## Reproduce the patch checks

Local Windows/Python 3.10 validation: the combined command below passed 200 tests
with one POSIX-mode skip and one third-party deprecation warning. A subsequent
25-test MCP/claim-gate run passed after pinning the subprocess to this source
checkout and adding the unnamed-comparison regression. Lint, the README artifact
gate, the local fixture benchmark, and benchmark-document synchronization passed.
These counts overlap and must not be added together as independent evidence.

```bash
python -m pytest tests/test_native_repair_consent.py tests/test_simulate_degraded_engine.py tests/test_receipt_store_writes.py tests/test_receipt_store_discovery.py tests/test_verify_readme_claims.py tests/test_cli_doctor.py tests/test_context_receipts.py tests/test_context_commit.py tests/test_receipt_path_containment.py tests/test_cli_compress_recover.py tests/test_recovery_store_trust.py tests/test_product_telemetry.py tests/test_mcp_protocol.py -q --tb=short --timeout=60
python scripts/verify_readme_claims.py
python bench/compare.py --check-regression --markdown bench_section.md
ruff check entroly/
```

Tests are maintainer-run engineering evidence, not independent certification or
proof of general answer quality. POSIX mode assertions require a POSIX runner;
Windows uses inherited ACLs. Existing CI runs the Python suite on Linux, including
the new permission and consent regressions. The benchmark workflow checks that the
generated fixture table matches `BENCHMARKS.md`. No paid model trial is part of this
remediation and no new savings or quality result is claimed.

## Draft correction request — not sent

Thank you for identifying concrete trust issues in Entroly. We have prepared a
focused remediation covering package-installation consent, recovery-file writes,
privacy disclosure, and unsupported README assurances. Please evaluate the linked
patch at its reviewed commit, keeping an unreleased patch separate from what users
can currently install.

One factual evidence addition: `CHANGELOG.md` existed at the revision you reviewed
(linked above). We also repaired stale release links. Please consider including
the changelog when revisiting the convention score.

We are not requesting removal of justified caveats: recovery files retain sensitive
input, independent task-quality replication remains incomplete, and the project
still depends on an individual maintainer. We do not claim a higher score or a
released fix until those states are independently observable.

## Rollback and release boundary

Before merging, require the PR checks to pass. Reverting the commit restores old
behavior; retain `ENTROLY_NO_SELF_HEAL=1` if reverting to a version with implicit
repair. File formats are unchanged and no data migration is required. Do not
delete recovery artifacts as part of a rollback. Publish a version-aligned patch
only after the release gates pass, then verify package installation and request
reassessment against that exact revision.
