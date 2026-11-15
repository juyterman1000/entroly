"""Does a recorded verification result still apply here, now?

A successor agent receives "the tests passed". That is history, not current
evidence, and two different questions are hidden in it:

* **Provenance:** who produced the result, by running what, and was it
  recorded by Entroly or only stated by an agent?
* **Applicability:** do the inputs the verification depended on still have the
  values they had when it ran?

This module answers both and keeps them apart. ``run_and_record`` runs a
verification command itself, so the exit code is observed rather than stated.
It traces what every Python process of the command reads: files, the
directories it lists, paths it probes, environment variables, the interpreter,
and the files behind every loaded module. It binds the result to digests of
exactly those inputs. ``assess`` recomputes the digests in the current
workspace and environment and returns one of three decisions:

* ``reuse``: every bound input is unchanged and the trace was complete;
* ``revalidate``: at least one bound input changed (the changes are listed);
* ``unknown``: applicability cannot be established. The trace was incomplete,
  an input could not be read, or the record failed its integrity check. Treat
  it as revalidate; never as reuse.

Soundness assumptions, stated rather than implied:

* The verification outcome is a deterministic function of the bound inputs. Clock time,
  randomness, network responses and hardware are not bound.
* Native code is opaque. File reads and ``getenv`` calls made from C
  extensions or by non-Python child processes are not observed. A child
  process that is not traced makes the record ``unknown``, but in-process
  native reads cannot be detected.
* Tool caches (``__pycache__``, ``.pytest_cache``, ``.mypy_cache``,
  ``.ruff_cache``, ``.hypothesis``, ``.coverage*``, ``.testmondata*``), temporary
  directories and Entroly's own state are excluded. A stored Hypothesis example
  or a pytest ``--lf`` selection is therefore not an input.
* A path probed with ``os.stat`` relative to an open directory descriptor is
  not resolved. Only temporary-directory cleanup does this in practice.

The record stores digests, never file contents, environment values or
command output. Environment values are hashed with a per-record salt.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "entroly.verification-evidence.v1"
PROVENANCE = "entroly-traced-run"
REUSE = "reuse"
REVALIDATE = "revalidate"
UNKNOWN = "unknown"

_CACHE_PARTS = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
})
_CACHE_PREFIXES = (".coverage", ".testmondata")
_PSEUDO_ROOTS = ("/proc", "/sys", "/dev")
_PACKAGE_METADATA_SUFFIXES = (".pth", ".dist-info", ".egg-info", ".egg-link")
_MAX_INPUTS = 50_000
_MAX_HASH_BYTES = 1 << 30
_MAX_TRACE_BYTES = 64 << 20
_MAX_REPORTED_CHANGES = 64
_ABSENT = "absent"
_STARTUP_ENV = (
    "PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "PYTHONNOUSERSITE", "PYTHONUSERBASE",
    "PYTHONWARNINGS", "PYTHONHASHSEED", "PYTHONOPTIMIZE", "PYTHONDEVMODE", "PYTHONUTF8",
    "PYTHONIOENCODING", "PYTHONPLATLIBDIR", "PYTHONINSPECT", "PYTHONSTARTUP",
)


class EvidenceError(RuntimeError):
    """The evidence record could not be created, stored or loaded."""


# ── canonical encoding ───────────────────────────────────────────────


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8", "surrogatepass")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def record_digest(record: Mapping[str, Any]) -> str:
    """Digest of the complete record, as bound into a handoff offer."""
    return _sha256(_canonical(dict(record)))


def _body_digest(record: Mapping[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k not in ("record_id", "verification_record")}
    return _sha256(_canonical(body))


# ── observation of one input ─────────────────────────────────────────


def _file_digest(path: Path) -> str:
    try:
        st = path.stat()
    except FileNotFoundError:
        return _ABSENT
    except OSError:
        return "unreadable"
    if not os.path.isfile(path):
        return "dir" if path.is_dir() else "other"
    if st.st_size > _MAX_HASH_BYTES:
        return "unreadable"
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return "unreadable"
    return "sha256:" + digest.hexdigest()


def _probe_state(path: Path) -> str:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return _ABSENT
    except NotADirectoryError:
        return _ABSENT
    except OSError:
        return "unreadable"
    import stat as _stat

    if _stat.S_ISREG(st.st_mode):
        return "file"
    if _stat.S_ISDIR(st.st_mode):
        return "dir"
    return "other"


def _module_stem(name: str) -> str:
    return name.split(".", 1)[0]


def _metadata_entry(name: str, directory: Path) -> bool:
    """The names ``importlib.metadata`` reads when it indexes a directory."""
    low = name.lower()
    return low.endswith((".dist-info", ".egg-info")) or (
        low == "egg-info" and directory.name.lower().endswith(".egg")
    )


def _listing_digest(path: Path, consumer: str, import_names: frozenset[str]) -> str:
    try:
        entries = os.listdir(path)
    except FileNotFoundError:
        return _ABSENT
    except NotADirectoryError:
        return _ABSENT
    except OSError:
        return "unreadable"
    # Tool caches are excluded as inputs, so their names are not inputs either:
    # a cleaned __pycache__ or a fresh .pytest_cache changes nothing a run used.
    entries = [
        entry for entry in entries
        if entry not in _CACHE_PARTS and not entry.startswith(_CACHE_PREFIXES)
    ]
    if consumer == "import":
        # The import system's directory cache resolves module names; only an
        # entry that could become (or stop being) one of the names attempted
        # matters, plus anything that changes what is installed.
        entries = [
            entry for entry in entries
            if _module_stem(entry) in import_names
            or _module_stem(entry) == "__init__"
            or entry.endswith(_PACKAGE_METADATA_SUFFIXES)
        ]
    elif consumer == "metadata":
        entries = [entry for entry in entries if _metadata_entry(entry, path)]
    return _sha256("\n".join(sorted(entries)).encode("utf-8", "surrogatepass"))


def _env_digest(salt: str, key: str, environ: Mapping[str, str], secret: bytes = b"") -> str:
    """Salted, and keyed when a state-directory secret is available.

    The salt stored in the record only stops correlation across records; with
    it alone a short value (a PIN) can be brute-forced from the record. The
    key never leaves Entroly's state directory, so the record alone does not
    give the value away.
    """
    value = environ.get(key)
    if value is None:
        return _ABSENT
    message = (
        bytes.fromhex(salt) + key.encode("utf-8", "surrogatepass") + b"\x00"
        + value.encode("utf-8", "surrogatepass")
    )
    if secret:
        import hmac

        return "hmac-sha256:" + hmac.new(secret, message, hashlib.sha256).hexdigest()
    return _sha256(message)


def _resolve_executable(argv0: str, cwd: Path, environ: Mapping[str, str]) -> str | None:
    if os.sep in argv0 or (os.altsep and os.altsep in argv0):
        candidate = Path(argv0)
        if not candidate.is_absolute():
            candidate = cwd / candidate
        return str(candidate) if candidate.exists() else None
    return shutil.which(argv0, path=environ.get("PATH", os.defpath))


def _executable_digest(argv0: str, cwd: Path, environ: Mapping[str, str]) -> str:
    located = _resolve_executable(argv0, cwd, environ)
    if not located:
        return _ABSENT
    # The invoked path, not only its target: two virtual environments share
    # one interpreter binary but have different site-packages, and Python
    # finds its environment from the path it was started by.
    invoked = os.path.abspath(located)
    real = os.path.realpath(located)
    content = _file_digest(Path(real))
    if not content.startswith("sha256:"):
        return content
    where = _sha256(f"{invoked}\x00{real}".encode("utf-8", "surrogatepass"))
    return f"{where};{content}"


class _Context:
    """Everything an observation depends on besides the input itself."""

    def __init__(
        self,
        project_root: Path,
        cwd: Path,
        environ: Mapping[str, str],
        salt: str,
        import_names: frozenset[str],
        secret: bytes = b"",
    ) -> None:
        self.project_root = project_root
        self.cwd = cwd
        self.environ = environ
        self.salt = salt
        self.import_names = import_names
        self.secret = secret

    def path(self, scope: str, key: str) -> Path:
        return self.project_root / key if scope == "project" else Path(key)


def _observe(item: Mapping[str, Any], context: _Context) -> str:
    kind = item["kind"]
    scope = item.get("scope", "")
    key = item["key"]
    if kind == "file":
        return _file_digest(context.path(scope, key))
    if kind == "probe":
        return _probe_state(context.path(scope, key))
    if kind == "listing":
        return _listing_digest(context.path(scope, key), "full", context.import_names)
    if kind == "import-listing":
        return _listing_digest(context.path(scope, key), "import", context.import_names)
    if kind == "metadata-listing":
        return _listing_digest(context.path(scope, key), "metadata", context.import_names)
    if kind == "env":
        return _env_digest(context.salt, key, context.environ, context.secret)
    if kind == "executable":
        return _executable_digest(key, context.cwd, context.environ)
    if kind == "project-root":
        return _sha256(os.path.realpath(context.project_root).encode("utf-8", "surrogatepass"))
    return "unresolved"


_UNRESOLVED = frozenset({"unreadable", "unresolved"})


# ── tracing a run ────────────────────────────────────────────────────


def _tracer_source() -> str:
    from . import evidence_tracer

    return Path(evidence_tracer.__file__).read_text(encoding="utf-8")


def _excluded_roots(trace_dir: Path, extra: Sequence[Path]) -> list[str]:
    roots = [os.path.realpath(tempfile.gettempdir()), os.path.realpath(trace_dir)]
    roots.extend(os.path.realpath(path) for path in extra)
    if os.name == "posix":
        roots.extend(_PSEUDO_ROOTS)
    return roots


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _source_for_cache(path: str) -> str | None:
    """``pkg/__pycache__/mod.cpython-312[-tag].pyc`` -> ``pkg/mod.py`` if present."""
    parent, name = os.path.split(path)
    if os.path.basename(parent) != "__pycache__" or not name.endswith(".pyc"):
        return None
    stem = name.split(".", 1)[0]
    source = os.path.join(os.path.dirname(parent), stem + ".py")
    return source if os.path.exists(source) else None


def _is_cache(path: str) -> bool:
    parts = Path(path).parts
    if any(part in _CACHE_PARTS for part in parts):
        return True
    return Path(path).name.startswith(_CACHE_PREFIXES)


class _Trace:
    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.first_seen: dict[str, tuple[int, int]] = {}
        self.writes: set[str] = set()
        self.listings: set[str] = set()
        self.import_listings: set[str] = set()
        self.metadata_listings: set[str] = set()
        self.probes: set[str] = set()
        self.env: set[str] = set()
        self.env_enumerated = False
        self.imports: set[str] = set()
        self.interpreters: set[str] = set()
        self.versions: set[str] = set()
        self.distributions: dict[str, str] = {}
        self.processes: dict[int, dict[str, Any]] = {}
        self.reasons: set[str] = set()


def _parse_traces(trace_dir: Path) -> _Trace:
    trace = _Trace()
    total = 0
    for file in sorted(trace_dir.glob("trace-*.jsonl")):
        try:
            size = file.stat().st_size
        except OSError:
            trace.reasons.add("trace-unreadable")
            continue
        total += size
        if total > _MAX_TRACE_BYTES:
            trace.reasons.add("trace-too-large")
            break
        process: dict[str, Any] = {"pid": None, "ppid": None, "exit": False, "spawns": 0, "argv": None}
        try:
            lines = file.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
        except OSError:
            trace.reasons.add("trace-unreadable")
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                trace.reasons.add("trace-corrupt")
                continue
            kind = event.get("t")
            if kind == "start":
                process["pid"] = event.get("pid")
                process["ppid"] = event.get("ppid")
                if isinstance(event.get("argv"), list):
                    process["argv"] = [str(part) for part in event["argv"]]
                if isinstance(event.get("exe"), str):
                    trace.interpreters.add(event["exe"])
                if isinstance(event.get("version"), str):
                    trace.versions.add(event["version"])
            elif kind == "open" and isinstance(event.get("p"), str):
                (trace.writes if event.get("w") else trace.reads).add(event["p"])
                seen_as = event.get("s")
                if not event.get("w") and isinstance(seen_as, list) and len(seen_as) == 2:
                    trace.first_seen.setdefault(event["p"], (int(seen_as[0]), int(seen_as[1])))
            elif kind == "list" and isinstance(event.get("p"), str):
                consumer = event.get("c")
                if consumer == "import":
                    trace.import_listings.add(event["p"])
                elif consumer == "metadata":
                    trace.metadata_listings.add(event["p"])
                else:
                    trace.listings.add(event["p"])
            elif kind == "probe" and isinstance(event.get("p"), str):
                trace.probes.add(event["p"])
            elif kind == "env" and isinstance(event.get("k"), str):
                trace.env.add(event["k"])
            elif kind == "env-iter":
                trace.env_enumerated = True
            elif kind == "import" and isinstance(event.get("m"), str):
                trace.imports.add(event["m"])
            elif kind == "spawn":
                process["spawns"] += 1
            elif kind == "exit":
                process["exit"] = True
                if event.get("lost"):
                    trace.reasons.add("trace-events-lost")
                modules = event.get("modules")
                if isinstance(modules, dict):
                    trace.reads.update(v for v in modules.values() if isinstance(v, str))
                dists = event.get("dists")
                if isinstance(dists, dict):
                    trace.distributions.update(
                        {str(k): str(v) for k, v in dists.items()}
                    )
        if isinstance(process["pid"], int):
            trace.processes[process["pid"]] = process
        else:
            trace.reasons.add("trace-without-start")
    return trace


def _same_file(left: str | None, right: str | None) -> bool:
    return bool(left and right) and os.path.realpath(left) == os.path.realpath(right)


def _root_started_by_python(
    started: Sequence[str] | None, argv: Sequence[str], cwd: Path, environ: Mapping[str, str],
) -> bool:
    """Did the traced root process start as ``argv`` itself?

    A program that ``exec``s Python keeps its pid, so the root's start record
    alone cannot tell ``python check.py`` from ``run.sh`` ending in
    ``exec python check.py``. The interpreter's own argv can. Accepted forms:
    ``argv`` run directly by the interpreter it names, or ``argv[0]`` a script
    whose shebang started the interpreter, which then read the script.
    """
    if not started:
        return False
    expected = _resolve_executable(argv[0], cwd, environ)
    direct = _resolve_executable(started[0], cwd, environ)
    if _same_file(direct, expected) and list(started[1:]) == list(argv[1:]):
        return True
    return (
        len(started) >= 2
        and _same_file(str(cwd / started[1]) if not os.path.isabs(started[1]) else started[1], expected)
        and list(started[2:]) == list(argv[1:])
    )


def _completeness(
    trace: _Trace,
    root_pid: int,
    argv: Sequence[str] = (),
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    reasons = set(trace.reasons)
    if root_pid not in trace.processes:
        reasons.add("command-not-traced")
    elif argv and not _root_started_by_python(
        trace.processes[root_pid].get("argv"), argv, cwd or Path.cwd(), environ or os.environ,
    ):
        reasons.add("command-started-by-untraced-program")
    children: dict[int, int] = {}
    for process in trace.processes.values():
        parent = process.get("ppid")
        if isinstance(parent, int):
            children[parent] = children.get(parent, 0) + 1
        if not process.get("exit"):
            reasons.add("process-ended-without-exit-record")
    for pid, process in trace.processes.items():
        if process["spawns"] > children.get(pid, 0):
            reasons.add("untraced-subprocess")
    return sorted(reasons)


def _classify(path: str, project_real: str, excluded: Sequence[str]) -> tuple[str, str] | None:
    real = os.path.realpath(path)
    if _is_cache(real):
        return None
    if _under(real, project_real):
        # The project wins over an excluded root that contains it: a checkout
        # under the system temp directory must keep every input. Only an
        # excluded root inside the project (Entroly's own state, say) applies.
        if any(_under(real, root) and _under(root, project_real) for root in excluded):
            return None
        relative = os.path.relpath(real, project_real)
        return "project", "." if relative == "." else Path(relative).as_posix()
    if any(_under(real, root) for root in excluded):
        return None
    return "external", real


def _inputs_from_trace(
    trace: _Trace,
    *,
    project_root: Path,
    argv0: str,
    excluded: Sequence[str],
    startup_keys: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], int, int]:
    project_real = os.path.realpath(project_root)
    items: dict[tuple[str, str, str], dict[str, Any]] = {}
    written = {os.path.realpath(path) for path in trace.writes}
    self_modified = 0

    def add(kind: str, path: str) -> None:
        located = _classify(path, project_real, excluded)
        if located is None:
            return
        scope, key = located
        items[(kind, scope, key)] = {"kind": kind, "scope": scope, "key": key}

    reads: set[str] = set()
    for path in trace.reads:
        source = _source_for_cache(path)
        reads.add(source if source is not None else path)
    for path in sorted(reads):
        real = os.path.realpath(path)
        if real in written and _classify(real, project_real, excluded) is not None:
            self_modified += 1
            continue
        if os.path.isdir(real):
            add("probe", path)
        else:
            add("file", path)
    for path in sorted(trace.probes):
        add("probe", path)
    # A full listing subsumes the narrow ones for the same directory.
    for path in sorted(trace.listings):
        add("listing", path)
    for path in sorted(trace.import_listings - trace.listings):
        add("import-listing", path)
    for path in sorted(trace.metadata_listings - trace.listings):
        add("metadata-listing", path)
    # Variables the interpreter reads in C before any tracer runs, present
    # or not, plus any other PYTHON* variable that is set.
    startup_env = set(_STARTUP_ENV) | {k for k in startup_keys if k.startswith("PYTHON")}
    for key in sorted(set(trace.env) | startup_env):
        items[("env", "", key)] = {"kind": "env", "scope": "", "key": key}
    # ``.pth`` files run at start-up, before the tracer: bind their contents.
    for directory in sorted(trace.import_listings | trace.metadata_listings):
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            if name.endswith(".pth"):
                add("file", os.path.join(directory, name))
    for exe in sorted(trace.interpreters):
        add("file", exe)
    items[("executable", "", argv0)] = {"kind": "executable", "scope": "", "key": argv0}
    items[("project-root", "", "")] = {"kind": "project-root", "scope": "", "key": ""}
    excluded_outputs = sum(1 for path in written if _classify(path, project_real, excluded) is None)
    return [items[k] for k in sorted(items)], self_modified, excluded_outputs


def _stop_group(process: subprocess.Popen) -> None:
    """Stop the command and everything it started, then reap it."""
    if os.name == "posix":
        import signal

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    else:  # pragma: no cover - Windows has no process groups here
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _digest_and_tail(path: Path, tail: int = 4000) -> tuple[str, bytes]:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
        size = handle.tell()
        handle.seek(max(0, size - tail))
        return "sha256:" + digest.hexdigest(), handle.read()


def _environment_for_child(
    environ: Mapping[str, str], site_dir: Path, trace_dir: Path
) -> dict[str, str]:
    from .evidence_tracer import TRACE_DIR_ENV

    child = dict(environ)
    original = child.get("PYTHONPATH", "")
    child["PYTHONPATH"] = str(site_dir) + (os.pathsep + original if original else "")
    child[TRACE_DIR_ENV] = str(trace_dir)
    return child


def _input_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment as the verification sees it, minus Entroly's own tracing."""
    from .evidence_tracer import TRACE_DIR_ENV

    return {k: v for k, v in environ.items() if k != TRACE_DIR_ENV}


def run_and_record(
    project_root: str | os.PathLike[str],
    argv: Sequence[str],
    *,
    name: str = "",
    cwd: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    work_dir: str | os.PathLike[str],
    excluded_roots: Sequence[str | os.PathLike[str]] = (),
    recorded_by: Mapping[str, Any] | None = None,
    repository_id: str = "",
    digest_secret: bytes = b"",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run ``argv`` with tracing and return ``(record, run_output)``.

    ``work_dir`` must be private to the caller. It holds the tracer and the
    trace files for the duration of the run and is emptied afterwards.
    ``run_output`` is never stored. It carries a bounded tail of the
    command's output for the caller to show.
    """
    if not argv or not all(isinstance(part, str) and part for part in argv):
        raise EvidenceError("the verification command must be a non-empty argument list")
    root = Path(project_root).resolve()
    run_cwd = Path(cwd).resolve() if cwd is not None else root
    if not (run_cwd == root or root in run_cwd.parents):
        raise EvidenceError("the verification must run inside the project")
    base_env = dict(os.environ if environ is None else environ)
    scratch = Path(work_dir) / f"run-{uuid.uuid4().hex}"
    site_dir = scratch / "site"
    trace_dir = scratch / "trace"
    for folder in (Path(work_dir), scratch, site_dir, trace_dir):
        folder.mkdir(mode=0o700, exist_ok=True)
        if os.name == "posix":
            os.chmod(folder, 0o700)
    try:
        (site_dir / "sitecustomize.py").write_text(_tracer_source(), encoding="utf-8")
        child_env = _environment_for_child(base_env, site_dir, trace_dir)
        started = time.time()
        out_path, err_path = scratch / "stdout", scratch / "stderr"
        # Output goes to private files, not pipes: a large log is not held in
        # memory, and a grandchild that keeps a pipe open cannot stall us.
        with open(out_path, "wb") as out_file, open(err_path, "wb") as err_file:
            try:
                process = subprocess.Popen(
                    list(argv),
                    cwd=str(run_cwd),
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=out_file,
                    stderr=err_file,
                    start_new_session=os.name == "posix",
                )
            except OSError as exc:
                raise EvidenceError(f"cannot start the verification command: {exc}") from exc
            try:
                process.wait(timeout=timeout_seconds)
            except BaseException as exc:  # timeout, Ctrl-C or SIGTERM: stop the whole group
                _stop_group(process)
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise EvidenceError("the verification command timed out; nothing was recorded") from exc
                raise
        duration_ms = int((time.time() - started) * 1000)
        stdout, stderr = _digest_and_tail(out_path), _digest_and_tail(err_path)
        trace = _parse_traces(trace_dir)
        excluded = _excluded_roots(scratch, [Path(p) for p in excluded_roots])
        reasons = _completeness(trace, process.pid, argv, run_cwd, base_env)
        inputs, self_modified, excluded_outputs = _inputs_from_trace(
            trace, project_root=root, argv0=argv[0], excluded=excluded,
            startup_keys=list(base_env),
        )
        # An input whose size or mtime moved between its first read and now
        # changed during the run; its post-run digest would bind the wrong state.
        for path, (size, mtime_ns) in trace.first_seen.items():
            if _classify(path, os.path.realpath(root), excluded) is None:
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            if (st.st_size, st.st_mtime_ns) != (size, mtime_ns):
                reasons = sorted(set(reasons) | {"input-changed-during-run"})
                break
        if self_modified:
            reasons = sorted(set(reasons) | {"input-modified-by-the-run"})
        if trace.env_enumerated:
            reasons = sorted(set(reasons) | {"environment-enumerated"})
        if len(inputs) > _MAX_INPUTS:
            raise EvidenceError("the verification read too many inputs to bind")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    salt = os.urandom(16).hex()
    import_names = frozenset(
        part for module in trace.imports for part in module.split(".") if part
    )
    context = _Context(root, run_cwd, _input_environment(base_env), salt, import_names, digest_secret)
    bound = []
    for item in inputs:
        digest = _observe(item, context)
        if digest in _UNRESOLVED:
            reasons = sorted(set(reasons) | {"input-unreadable"})
        bound.append(dict(item, digest=digest))
    relative_cwd = os.path.relpath(run_cwd, root)
    command = {"argv": list(argv), "cwd": Path(relative_cwd).as_posix()}
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "name": name or " ".join(argv)[:200],
        "command": command,
        "command_digest": _sha256(_canonical(command)),
        "result": {
            "exit_code": process.returncode,
            "verdict": "passed" if process.returncode == 0 else "failed",
            "duration_ms": duration_ms,
            "stdout_digest": stdout[0],
            "stderr_digest": stderr[0],
        },
        "recorded_at_ms": int(time.time() * 1000),
        "recorded_by": dict(recorded_by or {}),
        "provenance": PROVENANCE,
        "trace": {
            "complete": not reasons,
            "reasons": reasons,
            "processes": len(trace.processes),
            "excluded_outputs": excluded_outputs,
        },
        "observed": {
            "interpreter_versions": sorted(trace.versions),
            "distributions": dict(sorted(trace.distributions.items())),
        },
        "salt": salt,
        "env_digest": "hmac-sha256(state key)" if digest_secret else "sha256(salt)",
        "import_names": sorted(import_names),
        "inputs": bound,
        "inputs_digest": _sha256(_canonical(bound)),
    }
    record["record_id"] = "evidence:" + _body_digest(record).split(":", 1)[1][:32]
    record["verification_record"] = _verification_record(record, repository_id)
    output = {
        "stdout_tail": stdout[1].decode("utf-8", "replace"),
        "stderr_tail": stderr[1].decode("utf-8", "replace"),
    }
    return record, output


def _verification_record(record: Mapping[str, Any], repository_id: str) -> dict[str, Any]:
    """The Work Graph's own verification contract, bound to the input closure."""
    from .work_graph import create_verification_record

    return create_verification_record(
        repository_id=repository_id or "entroly-evidence",
        subject_id=f"command:{record['command_digest'].split(':', 1)[1][:32]}",
        subject_commitment=record["command_digest"],
        verified_repository_commitment=record["inputs_digest"],
        verdict=record["result"]["verdict"],
        observed_at_ms=int(record["recorded_at_ms"]),
        evidence_ids=[record["record_id"]],
    )


# ── integrity and storage ────────────────────────────────────────────


def verify_record(record: Any) -> list[str]:
    """Problems that make the record unusable as evidence; empty if none."""
    if not isinstance(record, Mapping):
        return ["record-not-an-object"]
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append("record-schema")
    try:
        if record.get("record_id") != "evidence:" + _body_digest(record).split(":", 1)[1][:32]:
            problems.append("record-id-mismatch")
        if record.get("inputs_digest") != _sha256(_canonical(record.get("inputs"))):
            problems.append("inputs-digest-mismatch")
        if record.get("command_digest") != _sha256(_canonical(record.get("command"))):
            problems.append("command-digest-mismatch")
    except (TypeError, ValueError, AttributeError):
        problems.append("record-malformed")
        return problems
    verification = record.get("verification_record")
    if not isinstance(verification, Mapping):
        problems.append("verification-record-missing")
    elif (
        verification.get("verified_repository_commitment") != record.get("inputs_digest")
        or verification.get("subject_commitment") != record.get("command_digest")
        or list(verification.get("evidence_ids") or []) != [record.get("record_id")]
        or verification.get("verdict") != (record.get("result") or {}).get("verdict")
    ):
        problems.append("verification-record-mismatch")
    return problems


def _store_dir(state_dir: str | os.PathLike[str]) -> Path:
    folder = Path(state_dir) / "evidence"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    if folder.is_symlink() or not folder.is_dir():
        raise EvidenceError("unsafe evidence directory")
    return folder


def _record_path(state_dir: str | os.PathLike[str], record_id: str) -> Path:
    if not isinstance(record_id, str) or not record_id.startswith("evidence:"):
        raise EvidenceError("evidence ids look like evidence:<32 hex>")
    tail = record_id.split(":", 1)[1]
    if len(tail) != 32 or any(ch not in "0123456789abcdef" for ch in tail):
        raise EvidenceError("evidence ids look like evidence:<32 hex>")
    return _store_dir(state_dir) / f"{tail}.json"


def save_record(state_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> Path:
    """Write once, atomically; an existing record with the same id is kept."""
    problems = verify_record(record)
    if problems:
        raise EvidenceError("refusing to store an inconsistent record: " + ", ".join(problems))
    target = _record_path(state_dir, str(record["record_id"]))
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical(dict(record)))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_record(state_dir: str | os.PathLike[str], record_id: str) -> dict[str, Any]:
    path = _record_path(state_dir, record_id)
    if path.is_symlink():
        raise EvidenceError("evidence record is a symbolic link")
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise EvidenceError(f"evidence record {record_id} is not available") from exc
    except OSError as exc:
        raise EvidenceError(f"evidence record {record_id} cannot be read: {exc}") from exc
    try:
        record = json.loads(raw.decode("utf-8", "surrogatepass"))
    except ValueError as exc:
        raise EvidenceError(f"evidence record {record_id} is corrupt") from exc
    if not isinstance(record, dict):
        raise EvidenceError(f"evidence record {record_id} is corrupt")
    return record


# ── assessment ───────────────────────────────────────────────────────


def provenance(record: Mapping[str, Any], *, bound_by: str = "") -> dict[str, Any]:
    """Who produced the result and how, kept separate from whether it applies."""
    result = record.get("result") if isinstance(record.get("result"), Mapping) else {}
    by = record.get("recorded_by") if isinstance(record.get("recorded_by"), Mapping) else {}
    return {
        "kind": record.get("provenance", ""),
        # What the record says about itself. Records are not signed: anyone who
        # can write Entroly's state directory can write one, so this is a claim,
        # never authentication. A host offer signature binds the record's
        # digest; it does not mean the host inspected the evidence.
        "observed_by_entroly": record.get("provenance") == PROVENANCE,
        "record_is_signed": False,
        "verdict": result.get("verdict", ""),
        "exit_code": result.get("exit_code"),
        "recorded_at_ms": record.get("recorded_at_ms"),
        "recorded_by_label": by.get("agent", ""),
        "recorded_by_is_authenticated": False,
        "authority_epoch": by.get("authority_epoch"),
        "bound_by": bound_by,
        "record_digest": record_digest(record) if isinstance(record, Mapping) else "",
    }


def assess(
    record: Mapping[str, Any],
    project_root: str | os.PathLike[str],
    *,
    environ: Mapping[str, str] | None = None,
    expected_digest: str = "",
    now_ms: int | None = None,
    digest_secret: bytes = b"",
) -> dict[str, Any]:
    """Decide reuse, revalidate or unknown for ``record`` in the current state."""
    root = Path(project_root).resolve()
    timestamp = int(now_ms if now_ms is not None else time.time() * 1000)
    problems = verify_record(record)
    if expected_digest and record_digest(record) != expected_digest:
        problems.append("record-not-the-one-handed-off")
    base = {
        "record_id": record.get("record_id") if isinstance(record, Mapping) else None,
        "name": record.get("name") if isinstance(record, Mapping) else None,
        "command": record.get("command") if isinstance(record, Mapping) else None,
    }
    if problems:
        return dict(base, decision=UNKNOWN, freshness=UNKNOWN, reasons=sorted(problems),
                    changed=[], changed_count=0, inputs_checked=0)
    trace = record["trace"]
    kinds = {item.get("kind") for item in record.get("inputs") or ()}
    if not {"executable", "file"} <= kinds:
        # A run that read no file and resolved no executable is not a traced
        # verification; it is a record someone wrote.
        return dict(base, decision=UNKNOWN, freshness=UNKNOWN, reasons=["closure-implausibly-small"],
                    changed=[], changed_count=0, inputs_checked=0)
    if not trace.get("complete"):
        return dict(base, decision=UNKNOWN, freshness=UNKNOWN,
                    reasons=["trace-incomplete", *trace.get("reasons", [])],
                    changed=[], changed_count=0, inputs_checked=0)
    command = record["command"]
    cwd = (root / command.get("cwd", ".")).resolve()
    context = _Context(
        root, cwd,
        _input_environment(os.environ if environ is None else environ),
        str(record["salt"]),
        frozenset(record.get("import_names") or ()),
        digest_secret if record.get("env_digest", "").startswith("hmac") else b"",
    )
    changed = []
    unresolved = []
    current = []
    for item in record["inputs"]:
        digest = _observe(item, context)
        current.append(dict(item, digest=digest))
        if digest in _UNRESOLVED:
            unresolved.append({"kind": item["kind"], "key": item["key"]})
        elif digest != item.get("digest"):
            changed.append({
                "kind": item["kind"], "scope": item.get("scope", ""), "key": item["key"],
                "was": "absent" if item.get("digest") == _ABSENT else "present",
                "now": "absent" if digest == _ABSENT else "present",
            })
    current_digest = "" if unresolved else _sha256(_canonical(current))
    from .work_graph import verification_freshness

    try:
        freshness = verification_freshness(
            record["verification_record"],
            current_repository_commitment=current_digest,
            now_ms=timestamp,
        )
    except Exception:  # an unverifiable contract is never evidence
        freshness = UNKNOWN
    if unresolved:
        decision, reasons = UNKNOWN, ["input-unresolved"]
    elif changed:
        decision, reasons = REVALIDATE, ["input-changed"]
    elif freshness == "current":
        decision, reasons = REUSE, ["all-inputs-unchanged"]
    else:
        decision, reasons = UNKNOWN, [f"verification-record-{freshness}"]
    return dict(
        base,
        decision=decision,
        freshness=freshness,
        reasons=reasons,
        changed=changed[:_MAX_REPORTED_CHANGES],
        changed_count=len(changed),
        unresolved=unresolved[:_MAX_REPORTED_CHANGES],
        inputs_checked=len(record["inputs"]),
    )


def summarize_inputs(record: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in record.get("inputs") or ():
        key = f"{item.get('kind')}:{item.get('scope') or 'none'}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def state_digest_secret(state_dir: str | os.PathLike[str]) -> bytes:
    """The per-state-directory key for environment digests (created once, 0600)."""
    path = _store_dir(state_dir) / ".digest-key"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as handle:
            handle.write(os.urandom(32))
    data = path.read_bytes()
    if len(data) != 32 or path.is_symlink():
        raise EvidenceError("the evidence digest key is damaged; remove it and record again")
    return data


__all__ = [
    "EvidenceError",
    "PROVENANCE",
    "REUSE",
    "REVALIDATE",
    "SCHEMA",
    "UNKNOWN",
    "assess",
    "load_record",
    "provenance",
    "record_digest",
    "run_and_record",
    "save_record",
    "state_digest_secret",
    "summarize_inputs",
    "verify_record",
]
