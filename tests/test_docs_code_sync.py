"""Docs-code sync gate: keep the published docs honest against the code.

Published docs rot when their code samples reference APIs that were later
renamed or removed — a documented ``from x import OldName`` that now raises on
import, wrong config fields, dead import paths. Entroly's differentiator is
verifiable honesty; this gate makes "every documented API actually exists" a CI
invariant so the docs cannot silently drift from the code.

Two checks, both high-signal and low-false-positive:

1. Every ``from entroly[...] import Name`` in the docs resolves to a real,
   importable symbol.
2. Every ``entroly <subcommand>`` shown in a code context is a real CLI
   subcommand (or a known command routed before argparse).
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOC_FILES = [ROOT / "README.md", ROOT / "PYPI_README.md", *sorted((ROOT / "docs").glob("*.md"))]

# Commands entroly routes to a local handler *before* argparse (so they do not
# appear in ``entroly --help``), but are real, shipped, and documented.
ROUTED_COMMANDS = frozenset({"memory"})

# `uvx --from entroly entroly` names the package then the console script; the
# token after `entroly ` there is the command name itself, not a subcommand.
NON_SUBCOMMAND_TOKENS = frozenset({"entroly"})

_IMPORT_RE = re.compile(r"^\s*from (entroly(?:\.[\w.]+)?) import ([A-Za-z0-9_,\s]+?)\s*(?:#.*)?$")
# `entroly <cmd>` only inside inline code (`...`) or a ``` fenced block ``` —
# never from prose, which would false-match ("the entroly server ...").
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_CMD_RE = re.compile(r"\bentroly ([a-z][a-z-]{2,})\b")


def _doc_texts() -> list[tuple[str, str]]:
    return [(p.name, p.read_text(encoding="utf-8", errors="replace")) for p in DOC_FILES if p.exists()]


def _documented_imports() -> list[tuple[str, str, str]]:
    """(doc_name, module, symbol) for every single-line entroly import."""
    out: list[tuple[str, str, str]] = []
    for name, text in _doc_texts():
        for line in text.splitlines():
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            module, names = m.group(1), m.group(2)
            if "(" in names or "\\" in names:
                continue  # parenthesized/continued multiline import — skip
            for sym in (s.strip() for s in names.split(",")):
                if sym.isidentifier():
                    out.append((name, module, sym))
    return out


def _code_spans(text: str) -> list[str]:
    """Inline-code spans plus fenced code blocks — the command-bearing contexts."""
    spans = _INLINE_CODE_RE.findall(text)
    in_fence = False
    buf: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if in_fence:
                spans.append("\n".join(buf))
                buf = []
            in_fence = not in_fence
            continue
        if in_fence:
            buf.append(line)
    return spans


def _real_cli_subcommands() -> set[str]:
    help_txt = subprocess.run(
        [sys.executable, "-m", "entroly", "--help"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    m = re.search(r"\{([a-z0-9,\-]+)\}", help_txt)
    return set(m.group(1).split(",")) if m else set()


def test_documented_entroly_imports_resolve() -> None:
    """Every `from entroly import X` in the docs must import cleanly."""
    broken: list[str] = []
    seen: set[tuple[str, str]] = set()
    for doc, module, sym in _documented_imports():
        if (module, sym) in seen:
            continue
        seen.add((module, sym))
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the gate
            broken.append(f"{doc}: `from {module} import {sym}` -> {type(exc).__name__}: {exc}")
            continue
        if not hasattr(mod, sym):
            broken.append(f"{doc}: `from {module} import {sym}` -> {sym} is not exported by {module}")
    assert not broken, "Documented entroly imports no longer resolve (docs drifted from code):\n" + "\n".join(broken)
    assert seen, "expected to find documented entroly imports to check"


def test_documented_cli_subcommands_exist() -> None:
    """Every `entroly <subcommand>` shown in a code context must be real."""
    valid = _real_cli_subcommands() | ROUTED_COMMANDS | NON_SUBCOMMAND_TOKENS
    assert len(_real_cli_subcommands()) > 20, "CLI help parse failed — cannot validate subcommands"

    unknown: dict[str, str] = {}
    for name, text in _doc_texts():
        for span in _code_spans(text):
            for cmd in _CMD_RE.findall(span):
                if cmd not in valid:
                    unknown.setdefault(cmd, name)
    assert not unknown, (
        "Docs reference entroly subcommands that do not exist "
        "(add to ROUTED_COMMANDS if routed before argparse):\n"
        + "\n".join(f"  entroly {c}  (in {f})" for c, f in sorted(unknown.items()))
    )
