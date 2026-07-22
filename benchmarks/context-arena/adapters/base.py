"""Adapter interface: how a context layer plugs into Context Arena.

Every system under test — retrieval layer, compressor, memory stack, context
OS — implements this one class. Adapters see exactly the same task inputs;
everything else about the run (model, budgets, oracle) is arm-invariant.
"""

from __future__ import annotations

from typing import Any, Protocol


class ContextAdapter(Protocol):
    """The single integration point for a context system under test."""

    #: Stable identifier used in artifacts and reports.
    name: str

    def prepare_context(self, task: dict[str, Any], token_budget: int) -> str:
        """Return the context string this layer selects for the task.

        Parameters
        ----------
        task:
            ``workspace`` — path to the repo snapshot (fix reverted);
            ``description`` — the task statement (commit subject);
            ``test_command`` — the oracle command (do NOT run it here);
            ``test_files`` / ``source_files`` — the touched files.
        token_budget:
            Maximum tokens the returned context may cost. Exceeding it is
            recorded as a protocol violation for the run, not silently
            truncated.
        """
        ...


class NoContextAdapter:
    """The mandatory baseline: the model sees only the task description."""

    name = "no-context"

    def prepare_context(self, task: dict[str, Any], token_budget: int) -> str:  # noqa: ARG002
        return ""
