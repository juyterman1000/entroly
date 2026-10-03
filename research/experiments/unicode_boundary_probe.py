"""Locate the EXACT Entroly frame that raises UnicodeEncodeError in the hook.

A previous attempt proved only that this machine's console is cp1252 and that
U+2192 is unencodable there -- by crashing the probe's own print(). That
establishes the environment, not the failing boundary. `sys.stdin.read()` is a
decode operation and cannot raise UnicodeEncodeError, so the real frame is a
downstream write.

This probe calls the hook path directly with the swallowing handler in
cli.py:257 bypassed, and reports the full traceback plus every stream's state.

  python research/experiments/unicode_boundary_probe.py
  PYTHONIOENCODING=cp1252 python research/experiments/unicode_boundary_probe.py
"""
from __future__ import annotations

import json
import locale
import os
import pathlib
import sys
import traceback

# Four distinct character classes; do not assume a single cause.
CLASSES = {
    "arrow U+2192": "→",
    "box U+250C": "┌",
    "greek U+03A6": "Φ",
    "geq U+2265": "≥",
    "emoji U+1F9E0": "\U0001f9e0",
}

FIXTURE_PROMPT = (
    "ASCII\n"
    "arrow: →\n"
    "box:\n┌──────┐\n"
    "│ test │\n└──────┘\n"
    "math: Φ λ ≥ ≤\n"
    "emoji: \U0001f9e0\n"
)


def describe(name: str, stream: object) -> dict:
    return {
        "name": name,
        "type": type(stream).__name__,
        "module": type(stream).__module__,
        "encoding": getattr(stream, "encoding", None),
        "errors": getattr(stream, "errors", None),
        "isatty": stream.isatty() if hasattr(stream, "isatty") else None,
    }


def emit(obj: object) -> None:
    """Report without being the thing that crashes.

    The probe must survive a cp1252 stdout or it cannot describe one.
    """
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str)
    data = text.encode("utf-8", errors="backslashreplace")
    buf = getattr(sys.stdout, "buffer", None)
    if buf is not None:
        buf.write(data + b"\n")
        buf.flush()
    else:  # pragma: no cover
        sys.stdout.write(text + "\n")


def main() -> int:
    emit("=== STREAM STATE ===")
    emit([describe("stdin", sys.stdin),
          describe("stdout", sys.stdout),
          describe("stderr", sys.stderr)])
    emit({
        "defaultencoding": sys.getdefaultencoding(),
        "filesystemencoding": sys.getfilesystemencoding(),
        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING"),
        "PYTHONUTF8": os.environ.get("PYTHONUTF8"),
        "locale_preferred": locale.getpreferredencoding(False),
        "platform": sys.platform,
    })

    emit("")
    emit("=== PER-CLASS ENCODABILITY per stream encoding ===")
    rows = []
    for label, ch in CLASSES.items():
        row = {"class": label, "cp": hex(ord(ch[0]))}
        for sname in ("stdout", "stderr"):
            enc = getattr(getattr(sys, sname), "encoding", None)
            if not enc:
                row[sname] = "no-encoding"
                continue
            try:
                ch.encode(enc)
                row[sname] = "OK(%s)" % enc
            except UnicodeEncodeError as exc:
                row[sname] = "FAIL(%s): %s" % (enc, exc.reason)
        rows.append(row)
    emit(rows)

    emit("")
    emit("=== INVOKING HOOK PATH (cli.py handler bypassed) ===")
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": FIXTURE_PROMPT,
        "cwd": str(pathlib.Path(__file__).resolve().parents[2]),
    }
    raw = json.dumps(payload)

    try:
        from entroly.agent_activation import parse_hook_input, run_hook
    except Exception:
        emit("import failed:")
        emit(traceback.format_exc())
        return 1

    try:
        parsed = parse_hook_input(raw)
        emit("parse_hook_input: OK (prompt chars=%d)" % len(parsed.get("prompt", "")))
    except Exception:
        emit("parse_hook_input RAISED -- full traceback:")
        emit(traceback.format_exc())
        return 1

    try:
        result = run_hook(parsed, host="claude", token_budget=1200, max_files=200)
    except Exception:
        emit("run_hook RAISED -- full traceback:")
        emit(traceback.format_exc())
        frames = traceback.extract_tb(sys.exc_info()[2])
        ent = [f for f in frames if "entroly" in f.filename.replace("\\", "/")]
        emit("=== FIRST/LAST ENTROLY APPLICATION FRAME ===")
        if ent:
            emit({"first": {"file": ent[0].filename, "line": ent[0].lineno,
                            "func": ent[0].name, "code": ent[0].line},
                  "last": {"file": ent[-1].filename, "line": ent[-1].lineno,
                           "func": ent[-1].name, "code": ent[-1].line}})
        else:
            emit("no entroly frame present")
        return 1

    emit("run_hook: OK (did NOT raise)")
    ctx = (result.get("hookSpecificOutput") or {}).get("additionalContext", "")
    non_ascii = sorted({hex(ord(c)) for c in ctx if ord(c) > 127})
    emit({"additionalContext_chars": len(ctx),
          "distinct_non_ascii": len(non_ascii),
          "sample": non_ascii[:20]})

    # The protocol write is the other candidate boundary; exercise it as cli.py
    # does (json.dumps default ensure_ascii=True) and report, do not crash.
    wire = json.dumps(result, separators=(",", ":"), sort_keys=True)
    all_ascii = all(ord(c) < 128 for c in wire)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        wire.encode(enc)
        emit("protocol json encodable on stdout(%s): YES; ensure_ascii produced %s"
             % (enc, "pure ASCII" if all_ascii else "raw non-ASCII"))
    except UnicodeEncodeError as exc:
        emit("protocol json NOT encodable on stdout(%s): %s" % (enc, exc))

    # And the context-format path cli.py uses for host=kiro, which prints raw.
    try:
        from entroly.agent_activation import hook_context
        raw_ctx = hook_context(result)
        raw_ctx.encode(enc)
        emit("hook_context() encodable on stdout(%s): YES" % enc)
    except UnicodeEncodeError as exc:
        emit("hook_context() NOT encodable on stdout(%s): %s -- this is the "
             "kiro/context output path" % (enc, exc))
    except Exception as exc:
        emit("hook_context() unavailable: %r" % (exc,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
