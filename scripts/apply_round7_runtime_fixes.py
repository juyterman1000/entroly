"""Apply the reviewed Round-7 runtime fixes deterministically.

Temporary branch-only helper. It performs exact, fail-closed source transforms so
Linux and Windows CI can validate the same patch before it is committed.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace_once(text: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return updated


def _append_once(path: Path, marker: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n\n" + content.strip() + "\n", encoding="utf-8")


def patch_auto_index() -> None:
    path = ROOT / "entroly" / "auto_index.py"
    text = path.read_text(encoding="utf-8")
    if "def _terminate_process_tree(" in text:
        return
    text = text.replace(
        "import os\nimport subprocess\nimport tempfile\n",
        "import os\nimport signal\nimport subprocess\nimport tempfile\n",
        1,
    )
    replacement = '''def _terminate_process_tree(
    proc: subprocess.Popen[bytes], *, timeout: float = 1.0
) -> None:
    """Best-effort termination of a process and descendants, then reap it.

    The command is started in an isolated POSIX session or Windows process group.
    On POSIX, killing the process group also closes inherited descriptors held by
    grandchildren. On Windows, ``taskkill /T`` is the supported tree primitive;
    direct ``kill`` remains the fail-safe fallback.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(0.1, timeout),
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=max(0.1, timeout))
    except (subprocess.TimeoutExpired, OSError, ValueError):
        # The caller must still return. There are no pipe-reader threads or pipe
        # handles to leak because stdout is a temporary file, not PIPE.
        pass


def _run_git(args: list[str], project_dir: str, timeout: float = 10.0) -> str | None:
    """Run a git command with a hard wall-clock bound and no pipe-reader leak.

    ``Popen(..., stdout=PIPE).communicate(timeout=...)`` is unsafe on Windows:
    ``communicate`` owns background reader threads, and a descendant that keeps
    stdout inherited can leave those threads and handles alive after the direct
    child is killed. Capture into an anonymous temporary file instead. Waiting
    for the process is then independent of EOF, and every Python-owned resource
    closes deterministically on every path.
    """
    proc: subprocess.Popen[bytes] | None = None
    timeout = max(0.01, float(timeout))
    try:
        with tempfile.TemporaryFile(mode="w+b") as capture:
            proc = subprocess.Popen(
                args,
                cwd=project_dir,
                stdin=subprocess.DEVNULL,
                stdout=capture,
                stderr=subprocess.DEVNULL,
                env=_git_env(),
                start_new_session=os.name != "nt",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "git %s timed out after %ss in %s; terminating its process "
                    "tree and falling back to a filesystem walk",
                    args[1] if len(args) > 1 else "",
                    timeout,
                    project_dir,
                )
                _terminate_process_tree(proc)
                return None

            if returncode != 0:
                return None
            capture.flush()
            capture.seek(0)
            return capture.read().decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError, ValueError):
        return None
    finally:
        if proc is not None and proc.poll() is None:
            _terminate_process_tree(proc)


'''
    text = _replace_once(
        text,
        r"def _run_git\(args: list\[str\], project_dir: str, timeout: float = 10\.0\) -> str \| None:.*?(?=def _git_ls_files)",
        replacement,
        label="auto_index._run_git",
    )
    path.write_text(text, encoding="utf-8")


def patch_checkpoint() -> None:
    path = ROOT / "entroly" / "checkpoint.py"
    text = path.read_text(encoding="utf-8")
    # Idempotence must key on CODE, not prose. This previously keyed on a
    # docstring sentence, so simply rewording that docstring would make this
    # script regex-replace the whole _enforce_global_cap body with the older
    # round-7 version — silently reverting the hard-cap and FileNotFoundError
    # fixes — and the workflow would then commit and push that revert.
    if "def _checkpoint_owner(" in text:
        return
    if "Checkpoint cap is soft" in text or "_enforce_global_cap" in text:
        # Newer logic is already present in some form; never overwrite it.
        return
    replacement = '''    def _enforce_global_cap(self) -> None:
        """Bound historical checkpoints without destroying active recovery state.

        The configured ceiling is strict only for *prunable history*. Recovery
        points outrank disk-count policy. The pass protects:

        * every retained checkpoint owned by this running instance;
        * the newest checkpoint for each peer that is alive or whose liveness
          cannot be proved locally (different host or unknown filename shape);
        * the newest checkpoint across known-dead local peers, preserving one
          restart/crash frontier while allowing older abandoned runs to collapse.

        If protected frontiers alone exceed the ceiling, the cap becomes soft.
        Deleting a live peer's last checkpoint would violate the stronger resume
        invariant and is never an acceptable way to satisfy a storage target.
        """
        if self.max_total_checkpoints <= 0:
            return
        try:
            candidates = list(self.checkpoint_dir.glob("ckpt_*.json.gz"))
        except OSError:
            return

        entries: list[tuple[float, Path]] = []
        for checkpoint in candidates:
            try:
                entries.append((checkpoint.stat().st_mtime, checkpoint))
            except OSError:
                continue
        if len(entries) <= self.max_total_checkpoints:
            return
        entries.sort(key=lambda item: item[0], reverse=True)

        own_prefix = f"ckpt_{self.instance_id}_"
        own_host = self.instance_id.split("_", 1)[0]
        protected: set[Path] = {
            checkpoint
            for _mtime, checkpoint in entries
            if checkpoint.name.startswith(own_prefix)
        }
        seen_peer_instances: set[str] = set()
        newest_dead_frontier: Path | None = None

        for _mtime, checkpoint in entries:
            if checkpoint in protected:
                continue
            owner = self._checkpoint_owner(checkpoint.name)
            if owner is None:
                # Unknown ownership is not permission to delete a recovery point.
                protected.add(checkpoint)
                continue
            owner_id, host_hash, pid = owner
            if owner_id in seen_peer_instances:
                continue
            seen_peer_instances.add(owner_id)

            if host_hash != own_host or self._pid_is_alive(pid):
                protected.add(checkpoint)
            elif newest_dead_frontier is None:
                # Keep one newest historical frontier for restart/crash recovery.
                newest_dead_frontier = checkpoint

        if newest_dead_frontier is not None:
            protected.add(newest_dead_frontier)

        remaining = len(entries)
        for _mtime, checkpoint in reversed(entries):
            if remaining <= self.max_total_checkpoints:
                break
            if checkpoint in protected:
                continue
            try:
                checkpoint.unlink()
            except OSError:
                continue
            remaining -= 1

        if remaining > self.max_total_checkpoints:
            import logging

            logging.getLogger("entroly.checkpoint").warning(
                "Checkpoint cap is soft: %d protected recovery frontier(s) "
                "require %d files, above configured cap %d",
                len(protected),
                remaining,
                self.max_total_checkpoints,
            )

'''
    text = _replace_once(
        text,
        r"    def _enforce_global_cap\(self\) -> None:.*?(?=    def _globbed_newest_first)",
        replacement,
        label="checkpoint._enforce_global_cap",
    )
    old = '''    _AUTO_ID_RE = re.compile(r"^ckpt_[0-9a-f]{12}_(\d+)_")

    def _peer_pid(self, name: str) -> int:
        """Owning pid from an auto-generated `ckpt_<hex12>_<pid>_...` name, else 0."""
        match = self._AUTO_ID_RE.match(name)
        if not match:
            return 0  # unknown shape — never guessed
        try:
            return int(match.group(1))
        except ValueError:
            return 0
'''
    new = '''    _AUTO_ID_RE = re.compile(r"^ckpt_([0-9a-f]{12})_(\d+)_")

    @classmethod
    def _checkpoint_owner(cls, name: str) -> tuple[str, str, int] | None:
        """Return ``(instance_id, host_hash, pid)`` for an auto filename."""
        match = cls._AUTO_ID_RE.match(name)
        if not match:
            return None
        try:
            pid = int(match.group(2))
        except ValueError:
            return None
        host_hash = match.group(1)
        return f"{host_hash}_{pid}", host_hash, pid

    def _peer_pid(self, name: str) -> int:
        """Owning pid from an auto-generated checkpoint name, else 0."""
        owner = self._checkpoint_owner(name)
        return owner[2] if owner is not None else 0
'''
    if old not in text:
        raise RuntimeError("checkpoint owner parser: exact source block not found")
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    _append_once(
        ROOT / "tests" / "test_checkpoint_retention.py",
        "test_global_cap_preserves_a_live_peers_latest_recovery_frontier",
        '''def test_global_cap_preserves_a_live_peers_latest_recovery_frontier(
    tmp_path: Path, monkeypatch
):
    """A busy writer must not erase a quieter live peer's only resume point."""
    import os

    live_pid = os.getpid()
    monkeypatch.setattr(
        CheckpointManager,
        "_pid_is_alive",
        staticmethod(lambda pid: pid == live_pid),
    )
    live_paths = [
        _write(
            tmp_path,
            f"ckpt_{_HOST}_{live_pid}_{i:03d}.json.gz",
            age_s=10_000 - i,
        )
        for i in range(3)
    ]
    expected_frontier = max(live_paths, key=lambda path: path.stat().st_mtime)

    # Newer abandoned history creates enough pressure to delete the live peer's
    # entire history under the previous globally-newest-only implementation.
    for i in range(30):
        _write(
            tmp_path,
            f"ckpt_{_HOST}_{_DEAD_PID}_{i:03d}.json.gz",
            age_s=100 - i,
        )

    mgr = CheckpointManager(
        tmp_path,
        instance_id=f"{_HOST}_1",
        max_checkpoints=10,
        max_total_checkpoints=5,
    )
    mgr._prune_old_checkpoints()

    live_remaining = list(tmp_path.glob(f"ckpt_{_HOST}_{live_pid}_*.json.gz"))
    assert live_remaining == [expected_frontier], (
        "the global cap erased or duplicated a live peer's protected frontier: "
        f"{live_remaining}"
    )
    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) <= 5


def test_global_cap_is_soft_when_protected_frontiers_exceed_it(tmp_path: Path):
    """Recovery safety outranks a count cap for unclassifiable ownership."""
    for i in range(3):
        _write(tmp_path, f"ckpt_unknown-owner-{i}.json.gz", age_s=100 + i)
    mgr = CheckpointManager(
        tmp_path,
        instance_id=f"{_HOST}_1",
        max_total_checkpoints=1,
    )

    mgr._prune_old_checkpoints()

    assert len(list(tmp_path.glob("ckpt_*.json.gz"))) == 3, (
        "unknown ownership was treated as permission to destroy recovery state"
    )''',
    )
    _append_once(
        ROOT / "tests" / "test_git_discovery_cannot_hang.py",
        "test_run_git_captures_to_a_file_not_a_pipe",
        '''def test_run_git_captures_to_a_file_not_a_pipe(monkeypatch, tmp_path):
    """No PIPE means no Windows communicate reader thread or pipe-handle leak."""
    real_popen = subprocess.Popen
    seen_stdout = []

    def recording_popen(*args, **kwargs):
        seen_stdout.append(kwargs.get("stdout"))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr("entroly.auto_index.subprocess.Popen", recording_popen)
    result = _run_git(
        [sys.executable, "-c", "print('tracked.py')"],
        str(tmp_path),
        timeout=5,
    )

    assert result is not None and result.splitlines() == ["tracked.py"]
    assert seen_stdout
    assert seen_stdout[0] is not subprocess.PIPE
    assert hasattr(seen_stdout[0], "fileno"), "stdout must be file-backed"


def test_run_git_never_calls_communicate(monkeypatch, tmp_path):
    """Pin the design: waiting must not depend on inherited stdout reaching EOF."""

    def forbidden_communicate(*_args, **_kwargs):
        raise AssertionError("_run_git must not use pipe-backed communicate()")

    monkeypatch.setattr(subprocess.Popen, "communicate", forbidden_communicate)
    result = _run_git(
        [sys.executable, "-c", "print('ok')"],
        str(tmp_path),
        timeout=5,
    )
    assert result is not None and result.splitlines() == ["ok"]''',
    )


def main() -> None:
    patch_auto_index()
    patch_checkpoint()
    patch_tests()


if __name__ == "__main__":
    main()
