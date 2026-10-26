"""CLI for Entroly Memory OS.

Run through the main Entroly command:

    entroly memory remember "important lesson"
    entroly memory recall "what matters?"

Or through the standalone console script:

    entroly-memory remember "important lesson"
    entroly-memory recall "what matters?"

Module form is also supported:

    python -m entroly.memory_cli remember "important lesson"
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from .memory import MemoryOS

_LOCK_TIMEOUT_S = 30.0


def default_memory_path() -> Path:
    return Path(os.environ.get("ENTROLY_MEMORY", ".entroly/memory.json")).expanduser()


@contextlib.contextmanager
def _memory_file_lock(path: Path) -> Iterator[None]:
    """Serialize one command's load-modify-save of ``path`` across processes.

    Each command rewrites the whole file, so without this lock a concurrent
    command's write is silently replaced by a stale snapshot. The lock lives
    in a sidecar file because the memory file itself is replaced atomically.
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        delay = 0.001
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out after {_LOCK_TIMEOUT_S:.0f}s waiting for memory lock {lock_path}"
                    ) from exc
                time.sleep(delay)
                delay = min(0.05, delay * 2)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_or_new(path: Path) -> MemoryOS:
    if path.exists():
        return MemoryOS.load(path)
    return MemoryOS()


def cmd_remember(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_memory_path()
    with _memory_file_lock(path):
        mem = load_or_new(path)
        mem.remember(
            args.content,
            agent_id=args.agent,
            importance=args.importance,
            tier=args.tier,
            source=args.source,
            tags=args.tag or [],
            safety_policy=args.safety_policy,
        )
        mem.save(path)
    if args.json:
        print(json.dumps({"status": "stored", "path": str(path), "stats": mem.stats()}, indent=2))
    else:
        print(f"stored memory in {path}")
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_memory_path()
    if not path.exists():
        print(f"no memory file found at {path}", file=sys.stderr)
        return 1
    with _memory_file_lock(path):
        mem = MemoryOS.load(path)
        ctx = mem.recall(args.query, agent_id=args.agent, budget=args.budget, tier=args.tier)
        mem.save(path)  # persist recall reinforcement
    if args.json:
        print(json.dumps(ctx.receipt(), indent=2, ensure_ascii=False))
    else:
        rendered = ctx.as_text()
        print(rendered if rendered else "no memories selected")
        print(f"\nused {ctx.used_tokens}/{ctx.budget} tokens; omitted {len(ctx.omitted)}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_memory_path()
    if not path.exists():
        stats = MemoryOS().stats()
    else:
        stats = MemoryOS.load(path).stats()
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser() if args.file else default_memory_path()
    if not path.exists():
        print(f"no memory file found at {path}", file=sys.stderr)
        return 1
    with _memory_file_lock(path):
        mem = MemoryOS.load(path)
        mem.tick(args.tick)
        forgotten = mem.forget(args.threshold)
        mem.save(path)
    print(json.dumps({"forgotten": forgotten, "stats": mem.stats()}, indent=2))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    mem = MemoryOS(safety_policy=args.safety_policy)
    result = mem.scan_safety(args.content)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0 if result.allowed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="entroly memory", description="Entroly Memory OS CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    remember = sub.add_parser("remember", help="Store a memory")
    remember.add_argument("content")
    remember.add_argument("--agent", default="default")
    remember.add_argument("--importance", type=float, default=0.5)
    remember.add_argument("--tier", choices=["working", "episodic", "semantic"], default="working")
    remember.add_argument("--source", default="manual")
    remember.add_argument("--tag", action="append", default=[])
    remember.add_argument("--file", default=None)
    remember.add_argument("--safety-policy", choices=["block", "redact", "allow"], default=None)
    remember.add_argument("--json", action="store_true")
    remember.set_defaults(func=cmd_remember)

    recall = sub.add_parser("recall", help="Recall memories under a token budget")
    recall.add_argument("query")
    recall.add_argument("--agent", default="default")
    recall.add_argument("--budget", type=int, default=1200)
    recall.add_argument("--tier", choices=["working", "episodic", "semantic"], default=None)
    recall.add_argument("--file", default=None)
    recall.add_argument("--json", action="store_true")
    recall.set_defaults(func=cmd_recall)

    stats = sub.add_parser("stats", help="Show memory stats")
    stats.add_argument("--file", default=None)
    stats.add_argument("--json", action="store_true", help="Accepted for compatibility; stats always emits JSON")
    stats.set_defaults(func=cmd_stats)

    forget = sub.add_parser("forget", help="Advance time and forget weak memories")
    forget.add_argument("--tick", type=int, default=0)
    forget.add_argument("--threshold", type=float, default=None)
    forget.add_argument("--file", default=None)
    forget.set_defaults(func=cmd_forget)

    scan = sub.add_parser("scan", help="Scan candidate memory for secrets/PII/injection")
    scan.add_argument("content")
    scan.add_argument("--safety-policy", choices=["block", "redact", "allow"], default="block")
    scan.set_defaults(func=cmd_scan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
