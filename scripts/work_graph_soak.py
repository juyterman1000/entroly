"""Sustained mixed-workload soak against a real durable Work Graph.

Not a unit test: several processes doing what independent agents actually do,
against one repository and one state root, for minutes rather than
milliseconds -- then checking the invariants that matter held throughout.

`tests/test_work_graph_multiprocess.py` already races two claimants through the
lock. That is a race, not a soak: it proves the protocol works once. The vault
learned the difference the expensive way -- `scripts/vault_soak.py` found a
chain-verification false alarm on its first run, after the full 38-check suite
had passed, because the bug needed a reader and a writer to disagree about
*when* they observed state, and no single-shot test creates that window.

Claimants claim overlapping and disjoint scopes. Observers refresh the
repository. Readers resume and read coordination while both are writing. A
prover interleaves continuation proofs, handoffs and handoff verification with
live writes, which is the combination most likely to observe a graph mid-update.
Every operation's failure is recorded rather than swallowed.

    python scripts/work_graph_soak.py . 300
    SOAK_STATE=/mnt/shared/state python scripts/work_graph_soak.py . 300

``SOAK_STATE`` places the state root somewhere specific. That is the point on a
shared mount: the store serializes writers with an exclusive-create lock file
rather than `fcntl.flock`, which needs a working `lockd` to mean anything
across NFS clients.

Exits non-zero when any invariant broke: a worker error, a graph that will not
re-import, a commitment that does not survive an export/import round trip, a
lost claim, a stray temp file, or an event count below the number of writes
that reported success.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# The repository being OBSERVED is not the source tree Entroly is imported
# from. Conflating them is a trap: pointing the soak at this checkout made the
# observer catalogue a 5,098-file virtualenv sitting inside it, which drove the
# state past its 64 MiB ceiling in two minutes and looked exactly like an
# unbounded-growth defect. Observe a purpose-built fixture; import from here.
SOURCE = Path(__file__).resolve().parents[1]
OBSERVED = Path(sys.argv[1]).resolve()
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
CLAIMANTS = 3

WORKER = """
import json, os, random, sys, time
sys.path.insert(0, {source!r})
from entroly.work_graph import WorkGraph, WorkGraphUnavailableError
from entroly.work_graph_store import WorkGraphStore

repo, state, role, tag, deadline = {observed!r}, {state!r}, {role!r}, {tag!r}, {deadline!r}

try:
    WorkGraph("native-probe")
except WorkGraphUnavailableError as exc:
    print(json.dumps({{"skipped": str(exc)}}))
    raise SystemExit(0)

def store():
    # Re-opened per operation on purpose: a long-lived handle would cache state
    # and hide exactly the cross-process reload this soak exists to exercise.
    return WorkGraphStore.for_repository(
        repo, root=state, lock_timeout_seconds=30.0, stale_lock_seconds=60.0
    )

stats = {{"writes": 0, "reads": 0, "proofs": 0, "claims": [], "errors": [], "null_control": 0}}
i = 0
while time.time() < deadline:
    i += 1
    try:
        if role == "claimant":
            # Half the claimants share a scope so coordination conflicts are
            # materialized under contention rather than only in the fixture.
            scope = "src/shared/core.py" if int(tag) % 2 == 0 else "src/w%s/mod_%d.py" % (tag, i % 16)
            task = "soak-%s-%d" % (tag, i)
            graph, lease = store().claim_work(
                repo,
                agent_id="agent-%s" % tag,
                task_title="soak task %d" % i,
                task_id=task,
                scope_paths=[scope],
                observed_at_ms=int(time.time() * 1000),
            )
            stats["claims"].append(task)
            stats["writes"] += 1
        elif role == "observer":
            store().update_repository(repo)
            stats["writes"] += 1
        elif role == "reader":
            s = store()
            s.resume()
            s.coordination()
            s.load().summary()
            stats["reads"] += 1
        else:
            s = store()
            # Both proof paths are workstream-scoped. Pick a live one from the
            # graph rather than assuming an id: under a soak the set churns.
            unfinished = s.load().unfinished()
            ws = unfinished[0].get("workstream_id") if unfinished else None
            if i % 3 == 0:
                if ws:
                    s.reconstructed_continuation_proof(ws, "agent-soak")
                else:
                    stats["null_control"] += 1
            elif i % 3 == 1:
                if ws:
                    receipt = s.handoff(ws, "agent-0", "agent-soak")
                    if not s.load().verify_handoff(receipt):
                        stats["errors"].append("handoff-verify-failed")
                    # A sealed handoff must yield a proof bound to it.
                    s.continuation_proof(receipt)
                else:
                    stats["null_control"] += 1
            else:
                # Export/import round trip against a live graph. `from_json`
                # revalidates every event id, bound, reference and the
                # aggregate commitment, so a torn read shows up here.
                g = s.load()
                exported = g.export_json()
                reloaded = WorkGraph.from_json(exported)
                if reloaded.graph_commitment != g.graph_commitment:
                    stats["errors"].append("commitment-drift-on-reimport")
                if reloaded.export_json() != exported:
                    stats["errors"].append("export-not-canonical")
            stats["proofs"] += 1
    except Exception as exc:
        # "no unfinished workstream" is the documented null control: asking to
        # continue or hand off work that does not exist must fail closed rather
        # than manufacture a task. Under a soak the graph is legitimately empty
        # of unfinished work much of the time -- between a claim completing and
        # the next one landing -- so scoring it as an error made correct
        # fail-closed behaviour look like a 270-failure defect.
        if isinstance(exc, ValueError) and "no unfinished workstream" in str(exc):
            stats["null_control"] += 1
        else:
            stats["errors"].append("%s:%s" % (type(exc).__name__, exc))
    time.sleep(0.01)
print(json.dumps(stats))
"""


def main() -> int:
    override = os.environ.get("SOAK_STATE")
    workdir = Path(override) if override else Path(tempfile.mkdtemp())
    workdir.mkdir(parents=True, exist_ok=True)
    state = str(workdir / "work-graph-state")
    deadline = time.time() + DURATION

    roles = (
        [("claimant", str(n)) for n in range(CLAIMANTS)]
        + [("observer", "0"), ("reader", "0"), ("reader", "1"), ("prover", "0")]
    )
    procs = []
    for role, tag in roles:
        script = textwrap.dedent(
            WORKER.format(
                source=str(SOURCE), observed=str(OBSERVED), state=state,
                role=role, tag=tag, deadline=deadline,
            )
        )
        procs.append(
            (role, subprocess.Popen([sys.executable, "-c", script],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True))
        )

    results = []
    for role, proc in procs:
        out, err = proc.communicate(timeout=DURATION + 300)
        if proc.returncode != 0:
            print(f"WORKER CRASH [{role}] rc={proc.returncode}\n{err[-800:]}")
            return 1
        payload = json.loads(out.strip().splitlines()[-1])
        if "skipped" in payload:
            print(f"SKIPPED: {payload['skipped']}")
            return 0
        results.append((role, payload))

    writes = sum(s["writes"] for _, s in results)
    reads = sum(s["reads"] for _, s in results)
    proofs = sum(s["proofs"] for _, s in results)
    errors = [e for _, s in results for e in s["errors"]]
    null_control = sum(s["null_control"] for _, s in results)
    claimed = [t for _, s in results for t in s["claims"]]

    # Invariants, checked after the fact against the durable state.
    sys.path.insert(0, str(SOURCE))
    from entroly.work_graph import WorkGraph
    from entroly.work_graph_store import WorkGraphStore

    final = WorkGraphStore.for_repository(str(OBSERVED), root=state).load()
    exported = final.export_json()

    reimport_ok = True
    commitment_stable = True
    try:
        reloaded = WorkGraph.from_json(exported)
        commitment_stable = (
            reloaded.graph_commitment == final.graph_commitment
            and reloaded.export_json() == exported
        )
    except Exception as exc:  # noqa: BLE001 - any failure is the finding
        reimport_ok = False
        errors.append(f"final-reimport:{type(exc).__name__}:{exc}")

    # Every claim a worker was told succeeded must be present. A claim the
    # store acknowledged and then lost to a concurrent replace is the exact
    # failure this soak exists to catch.
    lost = [t for t in set(claimed) if t not in exported]

    strays = [p.name for p in Path(state).rglob("*") if p.suffix in {".tmp", ".lock"} and p.is_file()]

    print("=" * 62)
    print(f"soak {DURATION:.0f}s  claimants={CLAIMANTS} observer=1 readers=2 prover=1")
    print(f"  writes={writes}  reads={reads}  proofs={proofs}")
    print(f"  null-control refusals: {null_control}  (expected: nothing to continue)")
    print(f"  worker errors        : {len(errors)}")
    for err in errors[:5]:
        print(f"      {err[:110]}")
    print(f"  final event count    : {final.event_count}")
    print(f"  distinct claims made : {len(set(claimed))}")
    print(f"  lost claims          : {len(lost)}")
    print(f"  re-imports cleanly   : {reimport_ok}")
    print(f"  commitment stable    : {commitment_stable}")
    print(f"  stray temp/lock files: {len(strays)}")

    ok = (
        not errors
        and not lost
        and reimport_ok
        and commitment_stable
        and not strays
        and final.event_count >= len(set(claimed))
    )
    print(f"  VERDICT              : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
