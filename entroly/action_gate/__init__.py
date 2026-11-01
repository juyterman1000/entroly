"""Action Gate: check an agent's tool calls against policy before they run.

Rules are declarative data (``action_rules.yaml``); nothing a model writes is
executed. Facts come only from tool results. Every decision is recorded so it
can be replayed from the rule file and the recorded facts.

This package imports nothing from the rest of ``entroly`` and uses only
relative imports. The pre-execution hook runs before every tool call, and
importing ``entroly`` costs hundreds of milliseconds, so ``_launch.py`` loads
this directory by path and the hook never pays for the rest of the package.
"""

from __future__ import annotations

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION"]
