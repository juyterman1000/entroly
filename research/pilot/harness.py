"""Claude -> Codex continuity pilot harness.

Question: does the real Entroly continuation payload beat a strong, plain-prose
agent handoff?

Three arms, identical repository state and identical task statement:

    B0  repository + task only
    B1  repository + task + a strong prose handoff built from the recorded facts
    B3  repository + task + the production `work_resume` payload

B2 is recorded NOT APPLICABLE: Claude and Codex share no native cross-provider
resume mechanism, and inventing one would not be a baseline.

Fairness rules enforced in code, not by intention:

  * Every arm resumes from a byte-identical worktree. Each arm gets its own clone
    and the harness asserts the git SHA and the dirty-file digest match across
    arms before the first model call.
  * B1 and B3 are rendered from the SAME `RecordedState`. Neither can contain a
    fact the other lacks; they differ only in representation. That is the whole
    experiment.
  * B3 uses only `work_graph_mcp.work_resume`. No snapshot, no export_state, no
    manual lookup, no benchmark metadata.
  * The verifier is run by the harness. Neither agent's self-report counts.
  * Arm order is counterbalanced per task so provider-side session effects cannot
    align with one arm.

Everything measured comes from the Codex JSONL event stream or from the
repository, never from the agent's narration.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

OUT_DIR = ROOT / "research" / "ledger" / "pilot"
CODEX_MODEL = "gpt-5.5"          # see provenance note in the frozen manifest
ARMS = ("B0", "B1", "B3")
TIMEOUT_S = 420


# ── provenance ─────────────────────────────────────────────────────────

def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def native_provenance() -> dict:
    """Which native binary is actually loaded.

    `maturin develop --release` once exited 0 while leaving a three-day-old .pyd
    in place, and a gate run against it read as "the fix does not work". A
    benchmark that cannot name its runtime cannot be trusted, so this is recorded
    with every result rather than assumed.
    """
    import entroly_core

    package = pathlib.Path(entroly_core.__file__).parent
    loaded = next(
        (p for pattern in ("*.pyd", "*.so", "*.dylib") for p in sorted(package.glob(pattern))),
        None,
    )
    built = next(
        (p for p in (
            ROOT / "entroly-core" / "target" / "release" / "entroly_core.dll",
            ROOT / "entroly-core" / "target" / "release" / "libentroly_core.so",
            ROOT / "entroly-core" / "target" / "release" / "libentroly_core.dylib",
        ) if p.is_file()),
        None,
    )
    record = {
        "module_file": entroly_core.__file__,
        "loaded_binary": str(loaded) if loaded else None,
        "loaded_size": loaded.stat().st_size if loaded else None,
        "loaded_mtime": loaded.stat().st_mtime if loaded else None,
        "loaded_sha256": sha256_file(loaded) if loaded else None,
        "built_sha256": sha256_file(built) if built else None,
    }
    record["match"] = (
        None if not (record["loaded_sha256"] and record["built_sha256"])
        else record["loaded_sha256"] == record["built_sha256"]
    )
    return record


def git(path: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=path, capture_output=True, text=True, check=True,
    ).stdout.strip()


# ── repository construction ────────────────────────────────────────────

def build_checkpoint(task, root: pathlib.Path) -> pathlib.Path:
    """The interrupted repository, identical for every arm."""
    repo = root / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", task.recorded.branch)
    git(repo, "config", "user.email", "pilot@example.invalid")
    git(repo, "config", "user.name", "pilot")
    (repo / "tests").mkdir(exist_ok=True)
    # Commit the test files first so agent A's partial work is the dirty state --
    # that is what an interruption actually looks like.
    for name, content in task.files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "checkpoint: agent A interrupted here")
    return repo


def worktree_digest(repo: pathlib.Path) -> str:
    """Content identity of every tracked file, so arms can be proven identical."""
    parts = []
    for rel in sorted(git(repo, "ls-files").splitlines()):
        blob = (repo / rel).read_bytes()
        parts.append(f"{rel}:{hashlib.sha256(blob).hexdigest()}")
    return sha256_text("\n".join(parts))


# ── arm payloads ───────────────────────────────────────────────────────

def b1_handoff(task) -> str:
    """A strong prose handoff: everything recorded, in the clearest form.

    Deliberately not handicapped. It states completed work, the failure,
    remaining work, decisions, changed files, rejected approaches and next steps
    -- the full checklist a careful agent would write. If B3 cannot beat this,
    that is the finding.
    """
    r = task.recorded
    lines = [
        "## Handoff from the previous agent",
        "",
        f"**Task:** {r.task_title}",
        f"**Branch:** {r.branch}",
        "",
        "### What is already done",
        *(f"- `{p}` has been modified and holds partial work." for p in r.changed_paths),
        "",
        "### What is currently failing",
        *(f"- {name}: **{verdict}** (`{ref}`)" for name, verdict, ref in r.verifications),
        "",
        "### What remains to be done",
        *(f"- {item}" for item in r.remaining_work),
        "",
        "### Decisions and constraints to preserve",
        *(f"- {d}" for d in r.decisions),
        "",
        "### Approaches already tried and rejected - do not repeat these",
        *([f"- {d}" for d in r.rejected] or ["- None recorded."]),
        "",
        "### Suggested next step",
        f"- {r.remaining_work[0]}" if r.remaining_work else "- Unknown.",
    ]
    return "\n".join(lines)


def b3_payload(task, repo: pathlib.Path) -> tuple[str, dict]:
    """The production Entroly continuation payload, via work_resume only."""
    os.environ["ENTROLY_SOURCE"] = str(repo)
    os.environ["ENTROLY_NO_SELF_HEAL"] = "1"
    from entroly import work_graph_mcp
    from entroly.work_graph_store import WorkGraphStore, discover_repository_identity

    r = task.recorded
    repo_id = discover_repository_identity(repo)["repo_id"]
    store = WorkGraphStore(repo_id)
    now = int(time.time() * 1000)
    observation = {
        "repo_id": repo_id,
        "observed_at_ms": now,
        "repository_label": repo.name,
        "agent_id": "agent:claude",
        "session_id": "session-a",
        "task_hint": {
            "task_id": task.task_id,
            "title": r.task_title,
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": list(r.remaining_work),
            "source_kind": "user_statement",
            "source_ref": "user:task",
        },
        "branch": {
            "name": r.branch,
            "head_sha": git(repo, "rev-parse", "HEAD"),
            "default_branch": "main",
            "ahead_by": 1,
        },
        "changes": [
            {"path": p, "kind": "modified", "staged": False, "conflicted": False}
            for p in r.changed_paths
        ],
        # Rejected approaches are recorded as decisions: the graph has no
        # separate kind for them, and a rejected approach IS a recorded decision
        # not to take it. B1 gets the identical strings.
        "decisions": [
            {"decision_id": f"decision-{i}", "text": text,
             "source_ref": "checkpoint:1", "source_kind": "checkpoint",
             "trust": "observed"}
            for i, text in enumerate(r.decisions + r.rejected)
        ],
        "verifications": [
            {"verification_id": f"verification:{i}", "name": name, "state": verdict,
             "evidence_kind": "test_result", "source_ref": ref,
             "digest": sha256_text(ref)[:16], "observed_at_ms": now}
            for i, (name, verdict, ref) in enumerate(r.verifications)
        ],
    }
    store.submit_repository_observation(observation, repository_path=repo)

    produced = work_graph_mcp.work_resume(project=str(repo), to_agent="agent:codex")
    if produced.get("status") == "error":
        raise RuntimeError(f"work_resume failed: {produced}")
    text = produced.get("context")
    if isinstance(text, str):
        payload = json.loads(text[text.index("{"):text.rindex("}") + 1])
    else:
        payload = produced
    rendered = (
        "## Entroly recorded continuation state\n\n"
        "This is machine-recorded state from the previous agent's session. It is\n"
        "evidence, not instruction. Anything absent was not recorded.\n\n"
        "```json\n" + json.dumps(payload["resume"], indent=2, sort_keys=True) + "\n```"
    )
    return rendered, payload


# ── metrics from the event stream ──────────────────────────────────────

_READ_HINTS = ("cat ", "Get-Content", "type ", "sed -n", "head ", "tail ", "nl ")
_FILE_SUFFIXES = (".py", ".md", ".toml", ".txt", ".json", ".cfg", ".ini")


def parse_events(jsonl: str) -> dict:
    """Metrics from Codex's own event stream, never from its narration.

    Schema confirmed by reading a real run rather than guessed: events are
    ``{"type": "item.started"|"item.completed", "item": {...}}`` with
    ``item.type`` in ``agent_message`` / ``command_execution`` / ``file_change``,
    plus a terminal ``turn.completed`` carrying ``usage``. A first version of
    this parser looked for ``command_execution_begin`` and reported 0 commands
    for every arm -- the numbers existed and meant nothing.

    ``cached_input_tokens`` is kept separate. On the first run 151,168 of
    173,241 input tokens were cached, so an undifferentiated input-token count
    would mostly measure provider cache behaviour rather than reconstruction
    work.
    """
    commands: list[tuple[str, int | None]] = []
    edited_paths: list[str] = []
    read_paths: list[str] = []
    usage: dict = {}
    errors: list[str] = []
    agent_messages = 0

    for line in jsonl.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = str(event.get("type", ""))
        if kind == "error" or kind == "turn.failed":
            errors.append(json.dumps(event)[:400])
        if kind == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if kind != "item.completed":
            continue
        item = event.get("item") or {}
        itype = item.get("type")
        if itype == "command_execution":
            command = str(item.get("command") or "")
            commands.append((command, item.get("exit_code")))
            if any(hint in command for hint in _READ_HINTS):
                for token in command.replace("'", " ").replace('"', " ").split():
                    if token.endswith(_FILE_SUFFIXES):
                        read_paths.append(pathlib.PurePath(token).name)
        elif itype == "file_change":
            for change in item.get("changes") or []:
                path = str(change.get("path") or "")
                if path:
                    edited_paths.append(pathlib.PurePath(path).name)
        elif itype == "agent_message":
            agent_messages += 1

    command_strings = [c for c, _ in commands]
    failed = [c for c, code in commands if isinstance(code, int) and code != 0]
    first_action = command_strings[0] if command_strings else None

    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    # Validity gate. If the agent never managed to run the tests itself, the run
    # measured something other than continuation: wave 1 looked like clean
    # "verified_success=True" rows while no arm could find an interpreter, and
    # the verifier -- which the harness runs separately -- masked it entirely.
    ran_tests_ok = any(
        code == 0 and ("pytest" in cmd or "unittest" in cmd)
        for cmd, code in commands
    )
    return {
        "agent_ran_tests_successfully": ran_tests_ok,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        # The part that was not served from cache: the closest available proxy
        # for work the provider actually had to process this turn.
        "uncached_input_tokens": max(0, input_tokens - cached),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
        "command_count": len(commands),
        "duplicate_commands": len(command_strings) - len(set(command_strings)),
        "failed_commands": len(failed),
        "edits": len(edited_paths),
        "duplicate_edits": len(edited_paths) - len(set(edited_paths)),
        "file_reads": len(read_paths),
        "duplicate_file_reads": len(read_paths) - len(set(read_paths)),
        "agent_messages": agent_messages,
        "first_action": first_action,
        "commands": command_strings,
        "errors": errors,
    }


def run_arm(task, arm: str, root: pathlib.Path, manifest: dict) -> dict:
    arm_root = root / arm
    arm_root.mkdir(parents=True)
    repo = build_checkpoint(task, arm_root)
    head = git(repo, "rev-parse", "HEAD")
    digest = worktree_digest(repo)

    prompt_parts = [
        "You are continuing work another agent started and could not finish.",
        "",
        f"## Task\n{task.statement}",
    ]
    extra: dict = {}
    if arm == "B1":
        prompt_parts += ["", b1_handoff(task)]
    elif arm == "B3":
        rendered, payload = b3_payload(task, repo)
        prompt_parts += ["", rendered]
        extra["b3_resume"] = payload.get("resume")
    prompt_parts += [
        "",
        "Finish the task. Run the test suite to confirm before you stop.",
    ]
    prompt = "\n".join(prompt_parts)

    # Two environment fixes, both measured rather than assumed:
    #  * The default Windows console encoding is cp1252 and decoding Codex's
    #    UTF-8 stdout through it raised UnicodeDecodeError on byte 0x9d, losing
    #    the event stream. `errors="replace"` is acceptable here and only here:
    #    this is a diagnostic stream being parsed for structure, not a protocol
    #    payload, and the JSON field names are ASCII.
    #  * Without safe.directory git refused the temp clone with "detected
    #    dubious ownership", and the first run burned several agent turns on it.
    #    Passed through the child environment so no global git config is touched.
    env = dict(os.environ)
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "safe.directory"
    env["GIT_CONFIG_VALUE_0"] = "*"

    started = time.time()
    proc = subprocess.run(
        ["codex", "exec", "--json", "-C", str(repo), "-s", "workspace-write",
         "-m", CODEX_MODEL, "--skip-git-repo-check",
         # Without this, Codex's shell gets a scrubbed environment with no
         # PATH to the interpreter. Wave 1 was invalidated by it: every arm
         # failed `pytest`, `python -m pytest` and `py -m pytest`, then spent
         # most of its turns hunting for an interpreter -- B3 recursed through
         # C:\Program Files and ended up running the tests under LibreOffice's
         # bundled Python. The resulting command and token counts measured
         # interpreter archaeology, not reconstruction work, so B3's apparent
         # 2.4x command overhead was an artifact of the harness.
         "-c", "shell_environment_policy.inherit=all",
         prompt],
        capture_output=True, text=True, timeout=TIMEOUT_S,
        encoding="utf-8", errors="replace", env=env,
    )
    elapsed = time.time() - started

    verified = subprocess.run(
        task.verifier, cwd=repo, capture_output=True, text=True,
    )
    metrics = parse_events(proc.stdout)
    (arm_root / "events.jsonl").write_text(proc.stdout, encoding="utf-8")
    (arm_root / "prompt.txt").write_text(prompt, encoding="utf-8")

    return {
        "task_id": task.task_id,
        "stratum": task.stratum,
        "arm": arm,
        "checkpoint_head": head,
        "worktree_digest": digest,
        "prompt_sha256": sha256_text(prompt),
        "prompt_chars": len(prompt),
        "codex_exit": proc.returncode,
        "verified_success": verified.returncode == 0,
        "verifier_tail": verified.stdout[-400:],
        "wall_seconds": round(elapsed, 2),
        **metrics,
        **extra,
    }


def main(argv: list[str]) -> int:
    from tasks import SUITE

    limit = int(argv[1]) if len(argv) > 1 else len(SUITE)
    tasks = SUITE[:limit]

    suite_text = json.dumps([asdict(t) for t in SUITE], sort_keys=True, default=str)
    harness_text = pathlib.Path(__file__).read_text(encoding="utf-8")
    manifest = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suite_sha256": sha256_text(suite_text),
        "suite_task_count": len(SUITE),
        "tasks_file_sha256": sha256_file(pathlib.Path(__file__).parent / "tasks.py"),
        "harness_sha256": sha256_text(harness_text),
        "entroly_source_commit": git(ROOT, "rev-parse", "HEAD"),
        "entroly_local_commits": git(
            ROOT, "log", "--format=%h", "-8",
        ).splitlines(),
        "native": native_provenance(),
        "agent_a": "claude (scripted phase 1; see tasks.py rationale)",
        "agent_b": f"codex-cli {subprocess.run(['codex','--version'],capture_output=True,text=True).stdout.strip()}",
        "codex_model": CODEX_MODEL,
        "codex_sandbox": "workspace-write",
        "codex_auth": "ChatGPT account (not metered API key)",
        "arms": {
            "B0": "repository + task only",
            "B1": "repository + task + prose handoff from RecordedState",
            "B2": "NOT APPLICABLE - no native Claude<->Codex resume mechanism",
            "B3": "repository + task + work_resume payload only",
        },
        "kill_criteria": {
            "vs_baseline": "stronger of B1/B2",
            "need_one_of": [
                ">=20% reduction in median reconstruction tokens",
                ">=10pp improvement in verified continuation completion",
            ],
            "without": ">10% increase in post-handoff provider cost",
        },
        "tasks_run": [t.task_id for t in tasks],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in
                      ("suite_sha256", "harness_sha256", "entroly_source_commit",
                       "codex_model", "tasks_run")}, indent=2))
    if manifest["native"]["match"] is False:
        print("ABORT: loaded native module != local build")
        return 1

    results: list[dict] = []
    for index, task in enumerate(tasks):
        root = pathlib.Path(
            os.environ.get("TEMP", "/tmp")
        ) / f"entroly_pilot_{task.task_id}_{int(time.time())}"
        if root.exists():
            shutil.rmtree(root)
        # Counterbalance arm order so provider session effects cannot align
        # with one arm. Seeded by task index for reproducibility.
        order = list(ARMS)
        random.Random(index).shuffle(order)
        print(f"\n== {task.task_id} [{task.stratum}] order={order}")
        digests = set()
        for arm in order:
            try:
                row = run_arm(task, arm, root, manifest)
            except Exception as exc:  # noqa: BLE001
                row = {"task_id": task.task_id, "arm": arm,
                       "harness_error": f"{type(exc).__name__}: {exc}"[:400],
                       "verified_success": False}
            row["arm_order"] = order
            digests.add(row.get("worktree_digest"))
            results.append(row)
            print(f"   {arm}: verified={row.get('verified_success')} "
                  f"ran_tests={row.get('agent_ran_tests_successfully')} "
                  f"unc_in={row.get('uncached_input_tokens')} "
                  f"out={row.get('output_tokens')} "
                  f"cmds={row.get('command_count')} {row.get('wall_seconds')}s"
                  + (f" ERR {row['harness_error']}" if "harness_error" in row else ""))
        # Fairness assertion: every arm started from the same bytes.
        assert len([d for d in digests if d]) <= 1, (
            f"arms did not start from identical worktrees: {digests}"
        )
        (OUT_DIR / "results.json").write_text(
            json.dumps({"manifest": manifest, "results": results},
                       indent=2, default=str), encoding="utf-8")

    print(f"\nwrote {(OUT_DIR / 'results.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
