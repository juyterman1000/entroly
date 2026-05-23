"""
Vault Integrity Check
=====================
Validates that all 179 Obsidian belief files are properly connected:
  1. Frontmatter is parseable
  2. [[wikilinks]] resolve to known entities or symbols
  3. No contradictions (stale-vs-verified, confidence divergence)
  4. Coverage gaps against source tree
"""
from entroly.vault import VaultConfig, VaultManager
from entroly.verification_engine import VerificationEngine

# ── Bootstrap the vault with the correct config chain ──
cfg = VaultConfig(base_path=r"c:\Users\abhis\entroly\.entroly\vault")
vm  = VaultManager(config=cfg)
ve  = VerificationEngine(vault=vm)

# ── Run the full verification pass ──
report = ve.full_verification_pass()
d = report.to_dict()

print("=" * 60)
print("  Entroly Vault Integrity Report")
print("=" * 60)
print(f"  Beliefs checked:  {d['total_beliefs_checked']}")
print(f"  Verified:         {d['verified_count']}")
print(f"  Stale:            {d['stale_count']}")
print(f"  Low confidence:   {d['low_confidence_count']}")
print(f"  Mean confidence:  {d['mean_confidence']}")
print(f"  Contradictions:   {d['contradictions']}")
print(f"  Coverage gaps:    {d['coverage_gaps']}")
print("=" * 60)

if d["contradiction_details"]:
    print(f"\n  Contradictions ({d['contradictions']}):")
    for c in d["contradiction_details"]:
        print(f"    [{c['severity']:6s}] {c['entity']}: {c['description']}")

# ── Also run coverage gap analysis against the source tree ──
gaps = ve.coverage_gaps(r"c:\Users\abhis\entroly\entroly")
if gaps:
    print(f"\n  Coverage Gaps ({len(gaps)} source files without beliefs):")
    for g in gaps[:15]:
        print(f"    {g.file_path} -> suggested entity: {g.suggested_entity}")
    if len(gaps) > 15:
        print(f"    ... and {len(gaps) - 15} more")

if not d["contradiction_details"] and not gaps:
    print("\n  All wikilinks resolved. Vault is fully connected and consistent.")
elif not d["contradiction_details"]:
    print(f"\n  No contradictions. {len(gaps)} coverage gaps remain.")
