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

DOC_FILES = [
    ROOT / "README.md",
    ROOT / "PYPI_README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]

# Commands entroly routes to a local handler *before* argparse (so they do not
# appear in ``entroly --help``), but are real, shipped, and documented.
ROUTED_COMMANDS = frozenset({"memory", "routing"})

# `uvx --from entroly entroly` names the package then the console script; the
# token after `entroly ` there is the command name itself, not a subcommand.
NON_SUBCOMMAND_TOKENS = frozenset({"entroly"})

_IMPORT_RE = re.compile(
    r"^\s*from (entroly(?:\.[\w.]+)?) import ([A-Za-z0-9_,\s]+?)\s*(?:#.*)?$"
)
# `entroly <cmd>` only inside inline code (`...`) or a ``` fenced block ``` —
# never from prose, which would false-match ("the entroly server ...").
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_CMD_RE = re.compile(r"\bentroly ([a-z][a-z-]{2,})\b")
_ENV_VAR_RE = re.compile(r"\bENTROLY_[A-Z0-9_]+\b")
_MARKDOWN_LINK_TARGET_RE = re.compile(r"\]\([^\n)]*\)")
# Generated inventories and architecture docs legitimately name files such as
# `ENTROLY_VERIFIED_MAP.md`. Those are file literals, not configuration knobs.
# Mask only compact backticked path/file tokens that contain a conventional
# extension; assignments such as `ENTROLY_DIR=/tmp/x` therefore remain visible.
_MARKDOWN_FILE_LITERAL_RE = re.compile(
    r"`(?=[^`\n=]*\.[A-Za-z0-9][A-Za-z0-9._-]*`)[^`\n=]+`"
)


def _doc_texts() -> list[tuple[str, str]]:
    return [
        (p.name, p.read_text(encoding="utf-8", errors="replace"))
        for p in DOC_FILES
        if p.exists()
    ]


def _documented_imports() -> list[tuple[str, str, str]]:
    """(doc_name, module, symbol) for every single-line entroly import."""
    out: list[tuple[str, str, str]] = []
    for name, text in _doc_texts():
        for line in text.splitlines():
            match = _IMPORT_RE.match(line)
            if not match:
                continue
            module, names = match.group(1), match.group(2)
            if "(" in names or "\\" in names:
                continue  # parenthesized/continued multiline import — skip
            for symbol in (value.strip() for value in names.split(",")):
                if symbol.isidentifier():
                    out.append((name, module, symbol))
    return out


def _code_spans(text: str) -> list[str]:
    """Inline-code spans plus fenced code blocks — the command-bearing contexts."""
    spans = _INLINE_CODE_RE.findall(text)
    in_fence = False
    buffer: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if in_fence:
                spans.append("\n".join(buffer))
                buffer = []
            in_fence = not in_fence
            continue
        if in_fence:
            buffer.append(line)
    return spans


def _documented_env_vars(text: str) -> set[str]:
    """Return advertised env vars, excluding Markdown destinations/file names."""
    visible_text = _MARKDOWN_LINK_TARGET_RE.sub("]()", text)
    visible_text = _MARKDOWN_FILE_LITERAL_RE.sub("``", visible_text)
    return set(_ENV_VAR_RE.findall(visible_text))


def _real_cli_subcommands() -> set[str]:
    help_text = subprocess.run(
        [sys.executable, "-m", "entroly", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    match = re.search(r"\{([a-z0-9,\-]+)\}", help_text)
    return set(match.group(1).split(",")) if match else set()


def test_documented_entroly_imports_resolve() -> None:
    """Every `from entroly import X` in the docs must import cleanly."""
    broken: list[str] = []
    seen: set[tuple[str, str]] = set()
    for doc, module, symbol in _documented_imports():
        if (module, symbol) in seen:
            continue
        seen.add((module, symbol))
        try:
            imported = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 — report, don't crash the gate
            broken.append(
                f"{doc}: `from {module} import {symbol}` -> "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if not hasattr(imported, symbol):
            broken.append(
                f"{doc}: `from {module} import {symbol}` -> "
                f"{symbol} is not exported by {module}"
            )
    assert not broken, (
        "Documented entroly imports no longer resolve "
        "(docs drifted from code):\n" + "\n".join(broken)
    )
    assert seen, "expected to find documented entroly imports to check"


def test_documented_env_vars_are_wired() -> None:
    """Every ENTROLY_* env var named in the docs must be read somewhere in code.

    Guards against advertising configuration knobs that nothing consumes.
    """
    documented: dict[str, str] = {}
    for name, text in _doc_texts():
        for variable in _documented_env_vars(text):
            documented.setdefault(variable, name)
    assert documented, "expected documented ENTROLY_* variables"

    wired: set[str] = set()
    for py_file in (ROOT / "entroly").rglob("*.py"):
        wired.update(
            _ENV_VAR_RE.findall(py_file.read_text(encoding="utf-8", errors="replace"))
        )
    for rust_file in (ROOT / "entroly-core" / "src").rglob("*.rs"):
        wired.update(
            _ENV_VAR_RE.findall(rust_file.read_text(encoding="utf-8", errors="replace"))
        )

    unwired = {value: source for value, source in documented.items() if value not in wired}
    assert not unwired, (
        "Docs advertise ENTROLY_* env vars that no code reads:\n"
        + "\n".join(
            f"  {variable}  (in {source})"
            for variable, source in sorted(unwired.items())
        )
    )


def test_env_var_scanner_ignores_markdown_link_destinations() -> None:
    text = (
        "Use `ENTROLY_PORT` with the "
        "[verified map](architecture/ENTROLY_VERIFIED_MAP.md)."
    )
    assert _documented_env_vars(text) == {"ENTROLY_PORT"}


def test_env_var_scanner_ignores_backticked_file_literals() -> None:
    text = (
        "Use `ENTROLY_PORT`; inventory files include "
        "`docs/architecture/ENTROLY_VERIFIED_MAP.md` and "
        "`ENTROLY_WIN_MASTER_PROMPT.md`. Keep "
        "`ENTROLY_DIR=/tmp/entroly` configurable."
    )
    assert _documented_env_vars(text) == {"ENTROLY_PORT", "ENTROLY_DIR"}


def test_documented_cli_subcommands_exist() -> None:
    """Every `entroly <subcommand>` shown in a code context must be real."""
    real_subcommands = _real_cli_subcommands()
    valid = real_subcommands | ROUTED_COMMANDS | NON_SUBCOMMAND_TOKENS
    assert len(real_subcommands) > 20, (
        "CLI help parse failed — cannot validate subcommands"
    )

    unknown: dict[str, str] = {}
    for name, text in _doc_texts():
        for span in _code_spans(text):
            for command in _CMD_RE.findall(span):
                if command not in valid:
                    unknown.setdefault(command, name)
    assert not unknown, (
        "Docs reference entroly subcommands that do not exist "
        "(add to ROUTED_COMMANDS if routed before argparse):\n"
        + "\n".join(
            f"  entroly {command}  (in {source})"
            for command, source in sorted(unknown.items())
        )
    )


def test_current_tree_respects_external_name_policy() -> None:
    """The permanent repository-wide policy must pass without exclusions."""
    try:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_external_name_policy.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            # The scan digests every sliding window of every line in the tree.
            # At 60s this raised TimeoutExpired on a loaded machine, so the
            # failure said "timed out" and never named the offending file --
            # a gate that cannot tell "policy broken" from "machine slow".
            # The scan now runs in ~42s locally; this leaves real slack.
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - CI pathology
        raise AssertionError(
            "external-name policy scan did not finish in 300s. This is a "
            "performance failure of the scan, not evidence of a violation; "
            "do not 'fix' it by deleting files."
        ) from exc

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "external-name policy check passed" in completed.stdout
