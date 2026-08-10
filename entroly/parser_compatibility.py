"""Runtime compatibility checks for optional parser dependencies.

Entroly keeps parser-backed code intelligence optional, but an already-installed
registry can still be older than the version floor declared by the current
package.  That state must never look healthy: callers can continue through
bounded fallbacks, while diagnostics make the capability loss explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import re
import warnings

TREE_SITTER_LANGUAGE_PACK_DISTRIBUTION = "tree-sitter-language-pack"
TREE_SITTER_LANGUAGE_PACK_MIN_VERSION = "1.14.3"

_VERSION_RE = re.compile(r"^\s*(\d+)\.(\d+)\.(\d+)(.*)$")
_WARNED_INCOMPATIBLE_VERSIONS: set[str] = set()


@dataclass(frozen=True)
class ParserRegistryStatus:
    installed: bool
    version: str | None
    minimum_version: str
    compatible: bool
    detail: str


def _version_at_least(installed: str, minimum: str) -> bool:
    """Conservatively compare the stable release floor without extra deps."""
    installed_match = _VERSION_RE.match(installed)
    minimum_match = _VERSION_RE.match(minimum)
    if installed_match is None or minimum_match is None:
        return False
    installed_release = tuple(int(item) for item in installed_match.groups()[:3])
    minimum_release = tuple(int(item) for item in minimum_match.groups()[:3])
    if installed_release != minimum_release:
        return installed_release > minimum_release

    suffix = installed_match.group(4).strip().lower()
    if not suffix:
        return True
    # Local/post releases of the floor are newer than the stable floor.  Pre-
    # releases/dev releases of the same numeric release are conservatively
    # treated as below the supported floor.
    return suffix.startswith((".post", "post", "+"))


def language_pack_status() -> ParserRegistryStatus:
    """Return the installed parser-registry compatibility state."""
    try:
        installed = metadata.version(TREE_SITTER_LANGUAGE_PACK_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return ParserRegistryStatus(
            installed=False,
            version=None,
            minimum_version=TREE_SITTER_LANGUAGE_PACK_MIN_VERSION,
            compatible=True,
            detail="optional_not_installed",
        )
    except Exception:
        return ParserRegistryStatus(
            installed=True,
            version=None,
            minimum_version=TREE_SITTER_LANGUAGE_PACK_MIN_VERSION,
            compatible=False,
            detail="version_unreadable",
        )

    compatible = _version_at_least(
        installed,
        TREE_SITTER_LANGUAGE_PACK_MIN_VERSION,
    )
    return ParserRegistryStatus(
        installed=True,
        version=installed,
        minimum_version=TREE_SITTER_LANGUAGE_PACK_MIN_VERSION,
        compatible=compatible,
        detail="compatible" if compatible else "below_supported_floor",
    )


def warn_if_incompatible_language_pack() -> ParserRegistryStatus:
    """Warn once when parser breadth may silently degrade due to version drift."""
    status = language_pack_status()
    if not status.installed or status.compatible:
        return status
    key = status.version or "unknown"
    if key in _WARNED_INCOMPATIBLE_VERSIONS:
        return status
    _WARNED_INCOMPATIBLE_VERSIONS.add(key)
    version = status.version or "unknown"
    warnings.warn(
        "Entroly detected tree-sitter-language-pack "
        f"{version}, below the supported >="
        f"{status.minimum_version} floor. Parser-backed repository intelligence "
        "may have materially reduced language coverage. Upgrade with "
        "`python -m pip install -U 'entroly[code-intelligence]'`. Entroly will "
        "continue only through bounded fallbacks where parser evidence is missing.",
        RuntimeWarning,
        stacklevel=2,
    )
    return status


__all__ = [
    "ParserRegistryStatus",
    "TREE_SITTER_LANGUAGE_PACK_DISTRIBUTION",
    "TREE_SITTER_LANGUAGE_PACK_MIN_VERSION",
    "language_pack_status",
    "warn_if_incompatible_language_pack",
]
