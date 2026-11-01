"""Where the gate reads rules and writes facts and decisions.

Rules live in their own file rather than under a key in ``policies.yaml``: the
governance loader requires that file to hold a ``policies`` list, so a file
written only for the gate would make governance fail to load.

A rules file is never read from the working directory. Rule feedback is shown
to the model, and a cloned repository must not be able to supply it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

RULES_FILE_ENV = "ENTROLY_ACTION_RULES"


def state_dir() -> Path:
    """``$ENTROLY_DIR`` when set, else ``~/.entroly``."""
    configured = os.environ.get("ENTROLY_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path("~/.entroly").expanduser()


def rules_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Rule file, most specific first.

    1. an explicit path
    2. ``ENTROLY_ACTION_RULES``
    3. ``$ENTROLY_DIR/governance/action_rules.yaml``
    4. ``~/.entroly/governance/action_rules.yaml``

    Same precedence as the governance policy file, so both sit side by side.
    """
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get(RULES_FILE_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return state_dir() / "governance" / "action_rules.yaml"


def gate_dir() -> Path:
    return state_dir() / "action_gate"


def session_key(session_id: str) -> str:
    """Identity for a session on disk; never the raw id."""
    return hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path
