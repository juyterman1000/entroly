"""Record what a verification process reads, for ``entroly-work evidence run``.

This file is copied, unchanged, into a private directory as
``sitecustomize.py``. That directory is then put first on ``PYTHONPATH`` for
the verification command. Every Python process the command starts imports it
at startup and appends one JSON line per observation to its own file in
``ENTROLY_EVIDENCE_TRACE_DIR``. When imported as ``entroly.evidence_tracer`` it
does nothing.

Observations:

* ``open``: every path opened, and whether it was opened for writing
  (from the ``open`` audit event);
* ``list``: every directory listed, and by whom (``os.listdir`` and
  ``os.scandir`` audit events; the immediate caller is read from the stack).
  The import system's directory cache and ``importlib.metadata`` consume only
  some names, so their listings can be bound narrowly; any other caller's
  listing is bound in full;
* ``import``: every module name the import system was asked for;
* ``probe``: every path checked with ``os.stat`` or ``os.lstat``, which is how
  pytest looks for ``conftest.py`` and configuration files that may be absent;
* ``env``: every environment variable read through ``os.environ`` that the
  process had not set itself;
* ``spawn``: every process started, so untraced children can be counted;
* ``exit``: the files behind every loaded module and the distributions they
  came from, written by an ``atexit`` handler. A process that ends without
  it, for example through ``os._exit``, is reported as incomplete.

It uses only the standard library, because it runs inside the project's own
interpreter, where Entroly is usually not installed. It must never change the
behaviour of the process it observes. Every hook swallows its own errors, and
an observation that cannot be written makes the trace incomplete, never wrong.
"""

from __future__ import annotations

TRACE_DIR_ENV = "ENTROLY_EVIDENCE_TRACE_DIR"
# Audit events that change a path without opening it for writing; every path
# argument is recorded as written, so a file the run read and then replaced is
# never bound as an input.
_MUTATION_EVENTS = frozenset({
    "os.rename", "os.remove", "os.rmdir", "os.truncate", "os.link", "os.symlink",
    "os.chmod", "os.utime", "shutil.copyfile", "shutil.copytree", "shutil.move", "shutil.rmtree",
})
_SPAWN_EVENTS = frozenset({
    "subprocess.Popen",
    "os.posix_spawn",
    "os.exec",
    "os.spawn",
    "os.system",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
})
_IMPORT_FILLER = "<frozen importlib._bootstrap_external>"
# ``importlib.metadata.FastPath.children`` lists every sys.path directory to
# find distributions; it uses only the *.dist-info and *.egg-info names.
_METADATA_FILES = ("importlib/metadata/__init__.py", "importlib_metadata/__init__.py")


class _Unbound:
    """A callable that stays a plain function when stored on a class."""

    __slots__ = ("_function",)

    def __init__(self, function) -> None:
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)


def _install() -> None:  # pragma: no cover - runs only inside traced children
    import atexit
    import json
    import os
    import sys

    trace_dir = os.environ.get(TRACE_DIR_ENV, "")
    if not trace_dir:
        return
    try:
        fd = os.open(
            os.path.join(trace_dir, f"trace-{os.getpid()}-{os.urandom(6).hex()}.jsonl"),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_APPEND", 0),
            0o600,
        )
    except OSError:
        return

    from _thread import get_ident

    # Re-entrancy is guarded per thread. One flag shared by all threads made a
    # hook running in one thread silently drop events from the others.
    busy_threads: set[int] = set()
    lost = [False]
    seen: set[tuple[str, str]] = set()
    written_env: set[str] = set()
    getcwd = os.getcwd
    abspath = os.path.abspath
    fsdecode = os.fsdecode
    write = os.write
    dumps = json.dumps

    def emit(record: dict) -> None:
        try:
            line = dumps(record, separators=(",", ":"), ensure_ascii=False)
            write(fd, (line + "\n").encode("utf-8", "surrogatepass"))
        except Exception:
            lost[0] = True  # reported in the exit record; the trace becomes incomplete

    def as_path(value) -> str | None:
        if value is None or isinstance(value, int):
            return None
        try:
            text = fsdecode(value)
        except Exception:
            return None
        if not text:
            return None
        try:
            return abspath(text)
        except Exception:
            return None

    def once(kind: str, key: str) -> bool:
        marker = (kind, key)
        if marker in seen:
            return False
        seen.add(marker)
        return True

    def writes(mode, flags) -> bool:
        if isinstance(mode, str) and mode:
            return any(ch in mode for ch in "wax+")
        if isinstance(flags, int):
            mask = os.O_WRONLY | os.O_RDWR | getattr(os, "O_CREAT", 0) | getattr(os, "O_TRUNC", 0)
            return bool(flags & mask)
        return False

    def hook(event: str, args) -> None:
        if get_ident() in busy_threads:
            return
        busy_threads.add(get_ident())
        try:
            if event == "open":
                path = as_path(args[0])
                if path is not None:
                    mode = args[1] if len(args) > 1 else None
                    flags = args[2] if len(args) > 2 else None
                    kind = "write" if writes(mode, flags) else "read"
                    if once(kind, path):
                        record = {"t": "open", "p": path, "w": kind == "write"}
                        if kind == "read":
                            # Size and mtime as first read, so the recorder can
                            # tell an input that changed during the run.
                            try:
                                st = original_stat(path)
                                record["s"] = [st.st_size, st.st_mtime_ns]
                            except OSError:
                                pass
                        emit(record)
            elif event in _MUTATION_EVENTS:
                for value in args[:2]:
                    path = as_path(value) if isinstance(value, (str, bytes)) or hasattr(value, "__fspath__") else None
                    if path is not None and once("write", path):
                        emit({"t": "open", "p": path, "w": True})
            elif event in ("os.listdir", "os.scandir"):
                target = args[0] if args else None
                path = as_path(target if target is not None else getcwd())
                if path is not None:
                    # Only the immediate caller counts. Module code that lists a
                    # directory while being imported also has import frames
                    # further up, and its listing must stay a full one.
                    consumer = "full"
                    frame = sys._getframe(1)
                    if frame is not None:
                        code = frame.f_code
                        if code.co_filename == _IMPORT_FILLER and code.co_name == "_fill_cache":
                            consumer = "import"
                        elif code.co_name == "children" and code.co_filename.replace("\\", "/").endswith(
                            _METADATA_FILES
                        ):
                            consumer = "metadata"
                    if once("list-" + consumer, path):
                        emit({"t": "list", "p": path, "c": consumer})
            elif event == "import":
                name = args[0] if args else None
                if isinstance(name, str) and once("import", name):
                    emit({"t": "import", "m": name})
            elif event in _SPAWN_EVENTS:
                emit({"t": "spawn", "e": event})
        except Exception:
            lost[0] = True
        finally:
            busy_threads.discard(get_ident())

    original_stat = os.stat
    original_lstat = os.lstat

    def probe(path) -> None:
        if get_ident() in busy_threads:
            return
        busy_threads.add(get_ident())
        try:
            text = as_path(path)
            if text is not None and once("probe", text):
                emit({"t": "probe", "p": text})
        except Exception:
            lost[0] = True
        finally:
            busy_threads.discard(get_ident())

    def traced_stat(path, *args, **kwargs):
        if kwargs.get("dir_fd") is None:  # relative to an open directory: not resolvable
            probe(path)
        return original_stat(path, *args, **kwargs)

    def traced_lstat(path, *args, **kwargs):
        if kwargs.get("dir_fd") is None:
            probe(path)
        return original_lstat(path, *args, **kwargs)

    original_access = os.access

    def traced_access(path, *args, **kwargs):
        if kwargs.get("dir_fd") is None:
            probe(path)
        return original_access(path, *args, **kwargs)

    environ_class = type(os.environ)
    original_getitem = environ_class.__getitem__
    original_setitem = environ_class.__setitem__
    original_delitem = environ_class.__delitem__
    original_iter = environ_class.__iter__

    def env_key(key) -> str | None:
        if isinstance(key, bytes):
            try:
                return key.decode("utf-8", "surrogateescape")
            except Exception:
                return None
        return key if isinstance(key, str) else None

    def traced_getitem(self, key):
        if get_ident() not in busy_threads:
            name = env_key(key)
            if name is not None and name not in written_env and name != TRACE_DIR_ENV:
                busy_threads.add(get_ident())
                try:
                    if once("env", name):
                        emit({"t": "env", "k": name})
                finally:
                    busy_threads.discard(get_ident())
        return original_getitem(self, key)

    def traced_setitem(self, key, value):
        name = env_key(key)
        if name is not None:
            written_env.add(name)
        return original_setitem(self, key, value)

    def traced_delitem(self, key):
        name = env_key(key)
        if name is not None:
            written_env.add(name)
        return original_delitem(self, key)

    def traced_iter(self):
        if get_ident() not in busy_threads:
            busy_threads.add(get_ident())
            try:
                if once("env-iter", ""):
                    emit({"t": "env-iter"})
            finally:
                busy_threads.discard(get_ident())
        return original_iter(self)

    def at_exit() -> None:
        busy_threads.add(get_ident())
        try:
            modules = {}
            for name, module in list(sys.modules.items()):
                location = getattr(module, "__file__", None)
                if isinstance(location, str) and location:
                    modules[name] = abspath(location)
            distributions = {}
            try:
                from importlib import metadata

                tops = {name.partition(".")[0] for name in modules}
                mapping = metadata.packages_distributions()
                for top in sorted(tops):
                    for dist in mapping.get(top, ()):
                        try:
                            distributions[dist] = metadata.version(dist)
                        except Exception:
                            distributions[dist] = ""
            except Exception:
                distributions = {"": "unresolved"}
            emit({"t": "exit", "modules": modules, "dists": distributions, "lost": lost[0]})
        except Exception:
            pass

    emit({
        "t": "start",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "exe": sys.executable,
        "version": sys.version,
        "cwd": getcwd(),
        "flags": [bool(sys.flags.isolated), bool(sys.flags.ignore_environment)],
        # The argv this interpreter was started with. A shell script ending
        # in ``exec python ...`` keeps its pid, so only this tells the
        # recorder that an untraced program ran first.
        "argv": list(getattr(sys, "orig_argv", sys.argv)),
    })
    stat_wrapper = _Unbound(traced_stat)
    lstat_wrapper = _Unbound(traced_lstat)
    access_wrapper = _Unbound(traced_access)
    # shutil.rmtree defends against symlink attacks only while os.stat is in
    # os.supports_dir_fd / supports_fd / supports_follow_symlinks; replacing the
    # function must not silently turn that defence off in the traced program.
    for capability in ("supports_dir_fd", "supports_fd", "supports_follow_symlinks"):
        members = getattr(os, capability, None)
        if isinstance(members, set):
            if original_stat in members:
                members.add(stat_wrapper)
            if original_lstat in members:
                members.add(lstat_wrapper)
    for capability in ("supports_dir_fd", "supports_effective_ids", "supports_follow_symlinks"):
        members = getattr(os, capability, None)
        if isinstance(members, set) and original_access in members:
            members.add(access_wrapper)
    # Install callable objects, not plain functions: a function stored as a
    # class attribute becomes a bound method. Python 3.10's pathlib keeps
    # ``stat = os.stat`` on its accessor class, so a function there would be
    # called with the accessor as an extra first argument and Path.exists()
    # would raise inside the traced program.
    os.stat = stat_wrapper
    os.lstat = lstat_wrapper
    os.access = access_wrapper
    environ_class.__getitem__ = traced_getitem
    environ_class.__setitem__ = traced_setitem
    environ_class.__delitem__ = traced_delitem
    environ_class.__iter__ = traced_iter
    sys.addaudithook(hook)
    # Only now: an interpreter without audit hooks (before 3.8) fails above and
    # never writes an exit record, so its trace is incomplete, not empty and
    # complete. Registered before the command's own handlers, so it runs last.
    atexit.register(at_exit)

    # Chain to the sitecustomize this one shadows, if there is one.
    here = os.path.dirname(abspath(__file__))
    try:
        from importlib import machinery, util

        search = [entry for entry in sys.path if abspath(entry or ".") != here]
        spec = machinery.PathFinder.find_spec("sitecustomize", search)
        if spec is not None and spec.loader is not None:
            chained = util.module_from_spec(spec)
            spec.loader.exec_module(chained)
    except Exception:
        pass


if __name__ == "sitecustomize":  # pragma: no cover - only as a copied sitecustomize
    _install()
