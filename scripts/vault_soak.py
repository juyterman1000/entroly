"""Sustained mixed-workload soak against a real vault.

Not a unit test: several processes doing what `entroly serve` and
`entroly compile` actually do, against a real source tree, for minutes rather
than milliseconds -- then checking the invariants that matter held throughout.

Writers append beliefs. Readers search and read them. A maintainer runs
retraction, backfill and chain verification concurrently with both. Every
operation's failure is recorded rather than swallowed.

    python scripts/vault_soak.py . 900
    SOAK_VAULT=/mnt/shared/vault python scripts/vault_soak.py . 900

``SOAK_VAULT`` places the vault somewhere specific, which is the point on a
shared mount: ledger appends are serialized by an exclusive-create lock file
because `fcntl.flock` needs a working `lockd` to mean anything across NFS
clients.

Measured, six processes each time, ledger records matching writes exactly with
an intact chain in every case: 900s on local disk (14,954 writes), 120s on SMB
(2,379), 120s on a real NFSv3 export (2,599), and 90s on the same export
mounted `nolock` (2,240). That last one is the case worth naming -- with
`local_lock=all` no lock reaches the server, which is how `flock` degrades on a
misconfigured mount, and the exclusive-create lock file carried the guarantee
on its own.

Run it against your own mount anyway. The point of the entry point is that a
deployment's storage is the one variable a test suite cannot stand in for.

Exits non-zero when any invariant broke: a worker error, an unreadable belief,
a stray temp file, an unparseable ledger record, a chain that does not verify,
or a ledger holding a different number of records than there were writes.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
WORKERS = 3

WORKER = """
import json, os, random, sys, time
sys.path.insert(0, {repo!r})
from entroly.vault import BeliefArtifact, VaultConfig, VaultManager
from entroly.vault_time import BeliefLedger

base, role, deadline = {base!r}, {role!r}, {deadline!r}
vault = VaultManager(VaultConfig(base_path=base))
stats = {{"writes": 0, "reads": 0, "maint": 0, "errors": []}}
i = 0
while time.time() < deadline:
    i += 1
    try:
        if role == "writer":
            vault.write_belief(BeliefArtifact(
                entity="%s-%d" % (role + {tag!r}, i),
                title="module %d" % i,
                body=("line %d\\n" % i) * random.randint(5, 60),
                sources=["src/mod_%d.py:%d" % (i % 50, i % 200)],
                source_root=".",
            ))
            stats["writes"] += 1
        elif role == "reader":
            vault.list_beliefs()
            vault.read_belief("%s-%d" % ("writer0", max(1, i % 50)))
            stats["reads"] += 1
        else:
            if i % 3 == 0:
                vault.mark_beliefs_ungrounded([base])
            elif i % 3 == 1:
                vault.backfill_source_roots([base])
            else:
                rep = BeliefLedger(vault._base).verify_chain()
                if rep["status"] != "intact":
                    stats["errors"].append("chain:" + str(rep.get("reason", rep)))
            stats["maint"] += 1
    except Exception as exc:
        stats["errors"].append("%s:%s" % (type(exc).__name__, exc))
    time.sleep(0.005)
print(json.dumps(stats))
"""


def main() -> int:
    override = os.environ.get("SOAK_VAULT")
    workdir = Path(override) if override else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    base = str(workdir / "vault")
    deadline = time.time() + DURATION

    procs = []
    roles = [("writer", str(n)) for n in range(WORKERS)] + [
        ("reader", "0"), ("reader", "1"), ("maint", "0")
    ]
    for role, tag in roles:
        script = textwrap.dedent(
            WORKER.format(repo=str(REPO), base=base, role=role,
                          deadline=deadline, tag=tag)
        )
        procs.append(
            (role, subprocess.Popen([sys.executable, "-c", script],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True))
        )

    results = []
    for role, proc in procs:
        out, err = proc.communicate(timeout=DURATION + 240)
        if proc.returncode != 0:
            print(f"WORKER CRASH [{role}] rc={proc.returncode}\n{err[-800:]}")
            return 1
        results.append((role, json.loads(out.strip().splitlines()[-1])))

    writes = sum(s["writes"] for _, s in results)
    reads = sum(s["reads"] for _, s in results)
    maint = sum(s["maint"] for _, s in results)
    errors = [e for _, s in results for e in s["errors"]]

    # Invariants, checked after the fact.
    sys.path.insert(0, str(REPO))
    from entroly.vault import _parse_frontmatter
    from entroly.vault_time import BeliefLedger

    beliefs = list((Path(base) / "beliefs").glob("*.md"))
    unreadable = [p.name for p in beliefs
                  if _parse_frontmatter(p.read_text(encoding="utf-8", errors="replace")) is None]
    strays = list((Path(base) / "beliefs").glob("*.tmp"))
    log = Path(base) / "ledger" / "beliefs.jsonl"
    records = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad_json = 0
    for line in records:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad_json += 1
    chain = BeliefLedger(Path(base)).verify_chain()

    print("=" * 62)
    print(f"soak {DURATION:.0f}s  writers={WORKERS} readers=2 maintainer=1")
    print(f"  writes={writes}  reads={reads}  maintenance={maint}")
    print(f"  worker errors      : {len(errors)}")
    for err in errors[:5]:
        print(f"      {err[:100]}")
    print(f"  belief files       : {len(beliefs)}")
    print(f"  unreadable beliefs : {len(unreadable)}")
    print(f"  stray temp files   : {len(strays)}")
    print(f"  ledger records     : {len(records)}  (writes={writes})")
    print(f"  unparseable records: {bad_json}")
    print(f"  chain              : {chain['status']} {chain.get('reason','')}")

    ok = (
        not errors and not unreadable and not strays and bad_json == 0
        and chain["status"] == "intact" and len(records) == writes
    )
    print(f"  VERDICT            : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
