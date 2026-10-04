"""Source-root and empty-index diagnostics for the MCP server.

A populated index can still point at the host application's directory; these
helpers keep corpus selection and empty-index guidance consistent across tools.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_PROJECT_ROOT_MARKERS: tuple[str, ...] = (
    ".git", ".hg", ".svn",
    "pyproject.toml", "setup.py", "requirements.txt",
    "package.json", "Cargo.toml", "go.mod",
    "pom.xml", "build.gradle", "Gemfile", "composer.json",
    ".entrolyignore",
)


@lru_cache(maxsize=64)
def _root_has_project_marker(source_root: str) -> bool:
    """True when ``source_root`` looks like something a person would index.

    Cheap filesystem probe, cached because every retrieval call asks.
    """
    try:
        root = Path(source_root)
        return any((root / marker).exists() for marker in _PROJECT_ROOT_MARKERS)
    except (OSError, ValueError):
        return False


def _source_root_is_suspicious(source_root: str) -> bool:
    """True when the indexed root was inherited rather than chosen, and does
    not look like a project.

    An MCP host launches this server with its *own* working directory. When
    ``ENTROLY_SOURCE`` is unset that directory becomes the index root, and
    ``auto_index`` falls back from ``git ls-files`` to walking the filesystem
    (auto_index.py: ``discovery = "walk"``). Walking an application bundle
    yields plenty of files, so ``ingested_count`` is healthy and every
    emptiness check passes -- while retrieval answers from a corpus that has
    nothing to do with the user's repository.

    Requiring an explicit ``ENTROLY_SOURCE`` would break the legitimate
    "cd into a repo and run" path, so this only fires when the root was both
    inherited *and* carries no project marker.
    """
    if os.environ.get("ENTROLY_SOURCE"):
        return False  # explicitly chosen by the operator; their call
    return not _root_has_project_marker(source_root)


def _source_root_guidance(source_root: str) -> dict[str, Any] | None:
    """Warn when a populated index is probably the wrong corpus."""
    if not _source_root_is_suspicious(source_root):
        return None
    return {
        "status": "suspicious_source_root",
        "message": (
            "This server indexed files, but its root was inherited from the "
            "host process and contains no project marker "
            f"({', '.join(_PROJECT_ROOT_MARKERS[:4])}, ...). Results may come "
            "from an unrelated directory such as the MCP client's application "
            "bundle rather than your repository."
        ),
        "resolve": [
            "Set ENTROLY_SOURCE to your repository root and restart the "
            "server (restart is required; the root is read once at startup).",
            "Confirm with get_stats that the fragment sources are your files.",
        ],
        "resolved_source_root": source_root,
    }


def _empty_context_guidance(
    ingested_count: int, source_root: str, *, tool: str = "optimize_context"
) -> dict[str, Any] | None:
    """Actionable diagnostic when a selection tool has nothing to select.

    A server that indexed no source files (commonly because its working
    directory is the MCP host's app dir, not the user's repo) previously
    returned ``selected: []`` with ``hallucination_risk: high`` and no
    explanation — indistinguishable, to the calling agent, from "no relevant
    context exists". Returns a guidance dict for the empty-session case, or
    ``None`` when fragments are present (a genuinely empty query match is not
    an error and gets no guidance).

    ``tool`` names the caller in the message. It is not cosmetic: the text is
    read by an agent deciding what to do next, so a hardcoded tool name sends
    the reader to inspect a tool it never called.
    """
    if ingested_count > 0:
        # A populated index is not proof of a *correct* index. The original
        # guard treated "something was ingested" as success, so a server rooted
        # at the MCP host's app bundle -- which walks up plenty of files --
        # passed silently and answered from the wrong corpus.
        return _source_root_guidance(source_root)
    # Do not assert a cause. This message previously said the empty index
    # "usually means the MCP server's working directory is not your project
    # root", and offered setting ENTROLY_SOURCE as the first remedy. Measured
    # against a freshly spawned `python -m entroly.server` on a 368-module
    # repository: with the working directory at the repository root *and*
    # ENTROLY_SOURCE set to it, recall_relevant still returned count 0. The
    # named cause was not the cause, and the first remedy did not work -- it
    # sent the reader to restart a server that would come back equally empty.
    #
    # Ingesting is what demonstrably populates the index (remember_fragment
    # followed by recall_relevant returned a correct hit in the same session),
    # so it is listed first. The root check stays, because a wrong root is a
    # real and separate failure, but it is offered as a thing to confirm rather
    # than as the diagnosis.
    return {
        "status": "no_codebase_indexed",
        "message": (
            f"{tool} returned nothing because this server has indexed no "
            "source files. An empty index is not evidence that the repository "
            "lacks the code; nothing has been read yet."
        ),
        "resolve": [
            "Ingest first: remember_fragment / smart_read / ingest, or "
            "read_source_file for a known path. This is what populates the "
            "index that recall_relevant searches.",
            "Then confirm the corpus is yours with get_stats: if fragment "
            "sources are not your files, set the ENTROLY_SOURCE environment "
            "variable (or the server's working directory) to your repository "
            "root and restart the server, since the root is read once at "
            "startup.",
        ],
        "resolved_source_root": source_root,
    }


def _ingested_fragment_count(engine: Any) -> int:
    """Fragments this server has indexed, preferring the native counter.

    Shared by every tool that must distinguish "nothing matched your query"
    from "nothing is indexed at all". Those answers look identical on the wire
    and lead the caller to opposite actions, so the check cannot be left to
    each call site to reimplement.
    """
    try:
        if getattr(engine, "_use_rust", False) and hasattr(
            engine._rust, "fragment_count"
        ):
            return int(engine._rust.fragment_count())
    except Exception:
        pass
    try:
        return int(getattr(engine, "_total_fragments_ingested", 0))
    except Exception:
        return 0
