"""Context Arena runner — self-contained CLI over the validated task miner.

Phase 1 (this file): mine and validate fail-to-pass tasks from any git repo.
Phase 2 (per PROTOCOL.md): run adapter arms + paired statistics.

Delegates to ``benchmarks.agentic_task_miner`` (tested: discovery, worktree
fail-to-pass validation) so the public harness and the internal experiments
can never drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Self-contained: resolve the repo root whether invoked from this directory,
# the repo root, or an arbitrary cwd.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.agentic_task_miner import main as _miner_main  # noqa: E402


def main() -> int:
    return _miner_main()


if __name__ == "__main__":
    sys.exit(main())
