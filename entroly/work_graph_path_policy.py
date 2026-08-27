"""Classification of worktree paths before they become recoverable evidence.

``git status --porcelain`` already omits git-ignored files, so a repository with
a well-maintained ``.gitignore`` looks clean. Real repositories are not that
tidy. A dependency tree that was never ignored, a ``.env`` written during
debugging, a private key dropped in the working directory for one test -- all
of these are untracked-and-not-ignored, so they arrive in recovery output as
ordinary changed paths.

Content is already safe: the digest layer never returns source bytes. The
filename is the leak. ``id_rsa`` and ``credentials.json`` disclose that a secret
exists and what kind, and a vendored dependency tree buries the two files the
agent actually touched under thousands it did not.

Three outcomes:

``ordinary``
    Reported unchanged.

``sensitive``
    Reported as a stable opaque id instead of a path. The agent still learns
    that one guarded file changed -- which matters, because a changed key is
    exactly the sort of thing that invalidates prior work -- without learning
    which. The id is a digest of the path, so it is stable across observations
    and comparable between them.

``generated``
    Omitted from paths, but counted. The receipt-honesty rule is that omitted
    evidence stays inspectable, so the count and the matched rules are always
    reported; a caller can tell "nothing else changed" from "9,000 vendored
    files changed and were dropped".

Deny-listing is not a security boundary and is not offered as one. It reduces
obvious disclosure from a surface whose whole job is to describe a working
directory it does not control. ``ENTROLY_WORK_GRAPH_SENSITIVE`` and
``ENTROLY_WORK_GRAPH_GENERATED`` extend either set for repositories with local
conventions this cannot know about.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass, field

# Directory components that indicate machine-produced or vendored content.
# Matched as a path segment, so `src/node_modules/x` matches but a file named
# `node_modules_notes.md` does not.
_GENERATED_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".gradle", ".mypy_cache", ".next", ".nuxt",
    ".parcel-cache", ".pytest_cache", ".ruff_cache", ".terraform", ".tox",
    ".turbo", ".venv", "bower_components", "coverage", "dist",
    "node_modules", "site-packages", "target", "vendor", "venv",
})

# Filename patterns for machine-produced files outside a generated directory.
# Lockfiles are deliberately absent: a changed lockfile is real work.
_GENERATED_FILES: tuple[str, ...] = (
    "*.min.js", "*.min.css", "*.map", "*.pyc", "*.pyo", "*.class", "*.o",
    "*.so", "*.dylib", "*.dll", "*_pb2.py", "*_pb2_grpc.py", "*.generated.*",
    "*.g.dart", "*.freezed.dart",
)

# Filename patterns that commonly carry credentials. Matched on the basename,
# case-insensitively.
_SENSITIVE_FILES: tuple[str, ...] = (
    ".env", ".env.*", "*.env",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.ppk",
    "credentials", "credentials.*", "*credentials.json",
    ".npmrc", ".pypirc", ".netrc", "_netrc", ".htpasswd",
    "secrets", "secrets.*", "*.secrets", "*secret*.json", "*secret*.yaml",
    "service-account*.json", "*serviceaccount*.json",
    ".aws", "*.kubeconfig", "kubeconfig",
)

ORDINARY = "ordinary"
SENSITIVE = "sensitive"
GENERATED = "generated"


def _extra(variable: str) -> tuple[str, ...]:
    raw = os.environ.get(variable, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass
class PathPolicyResult:
    """Classified paths plus the record of what was withheld and why."""

    paths: list[str] = field(default_factory=list)
    sensitive_ids: list[str] = field(default_factory=list)
    generated_omitted: int = 0
    sensitive_count: int = 0
    matched_rules: list[str] = field(default_factory=list)

    def as_disclosure(self) -> dict[str, object]:
        """The inspectable record of omission that accompanies the paths."""
        return {
            "generated_omitted": self.generated_omitted,
            "sensitive_withheld": self.sensitive_count,
            "matched_rules": sorted(set(self.matched_rules)),
        }


def sensitive_id(path: str) -> str:
    """Stable opaque identifier for a path that must not be disclosed.

    A digest rather than a counter so the same file yields the same id across
    observations and processes, which is what lets a caller notice that the
    *same* guarded file changed twice without ever learning its name.
    """
    digest = hashlib.sha256(path.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"sensitive:{digest[:16]}"


def classify(path: str) -> tuple[str, str]:
    """Return ``(classification, matched_rule)`` for one repository-relative path."""
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return ORDINARY, ""
    segments = normalized.split("/")
    basename = segments[-1].lower()

    # Sensitive wins over generated: a key inside a vendored tree is still a key,
    # and dropping it silently would be the worse failure.
    for pattern in _SENSITIVE_FILES + _extra("ENTROLY_WORK_GRAPH_SENSITIVE"):
        if fnmatch.fnmatch(basename, pattern.lower()):
            return SENSITIVE, f"sensitive:{pattern}"

    for segment in segments[:-1]:
        if segment.lower() in _GENERATED_DIRS:
            return GENERATED, f"dir:{segment}"
    # A directory-only path ends in "/" and loses its last segment above.
    if path.endswith("/") and segments[-1].lower() in _GENERATED_DIRS:
        return GENERATED, f"dir:{segments[-1]}"

    for pattern in _GENERATED_FILES + _extra("ENTROLY_WORK_GRAPH_GENERATED"):
        if fnmatch.fnmatch(basename, pattern.lower()):
            return GENERATED, f"generated:{pattern}"

    return ORDINARY, ""


def apply_policy(paths: list[str]) -> PathPolicyResult:
    """Split changed paths into disclosable, opaque, and omitted."""
    result = PathPolicyResult()
    seen_sensitive: set[str] = set()
    for path in paths:
        classification, rule = classify(path)
        if classification == ORDINARY:
            result.paths.append(path)
            continue
        result.matched_rules.append(rule)
        if classification == SENSITIVE:
            token = sensitive_id(path)
            result.sensitive_count += 1
            if token not in seen_sensitive:
                seen_sensitive.add(token)
                result.sensitive_ids.append(token)
        else:
            result.generated_omitted += 1
    result.paths.sort()
    result.sensitive_ids.sort()
    return result
