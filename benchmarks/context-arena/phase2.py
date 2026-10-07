"""Context Arena Phase 2: fail-closed multi-adapter execution primitives.

This module deliberately separates *measurement* from *claims*.  An adapter
that errors, exceeds its budget, or cannot be configured is BLOCKED; it is
never scored as a loss that another adapter can claim as a win.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Protocol


class ContextAdapter(Protocol):
    """Structural adapter contract; avoids package imports from hyphenated path."""

    name: str

    def prepare_context(self, task: dict[str, Any], token_budget: int) -> str:
        ...


class ArmStatus(str, Enum):
    OK = "ok"
    BLOCKED = "blocked"
    ERROR = "error"
    BUDGET_VIOLATION = "budget_violation"


@dataclass(frozen=True)
class ArmResult:
    task_id: str
    adapter: str
    status: ArmStatus
    context: str = ""
    estimated_tokens: int = 0
    detail: str = ""

    @property
    def valid_measurement(self) -> bool:
        return self.status is ArmStatus.OK

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["valid_measurement"] = self.valid_measurement
        return payload


def estimate_tokens(text: str) -> int:
    """Conservative dependency-free accounting used only for the context cap.

    Provider-observed token usage remains the source of truth for economic
    reporting.  The arena records this estimate so an adapter cannot silently
    exceed the shared context budget before the model call.
    """
    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def prepare_arm(
    adapter: ContextAdapter,
    task: dict[str, Any],
    token_budget: int,
) -> ArmResult:
    task_id = str(task.get("task_id") or task.get("id") or "unknown")
    name = str(getattr(adapter, "name", adapter.__class__.__name__))

    try:
        context = adapter.prepare_context(task, token_budget)
    except (ImportError, ModuleNotFoundError) as exc:
        return ArmResult(
            task_id=task_id,
            adapter=name,
            status=ArmStatus.BLOCKED,
            detail=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - failures are evidence
        return ArmResult(
            task_id=task_id,
            adapter=name,
            status=ArmStatus.ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(context, str):
        return ArmResult(
            task_id=task_id,
            adapter=name,
            status=ArmStatus.ERROR,
            detail=f"adapter returned {type(context).__name__}, expected str",
        )

    used = estimate_tokens(context)
    if used > token_budget:
        return ArmResult(
            task_id=task_id,
            adapter=name,
            status=ArmStatus.BUDGET_VIOLATION,
            context=context,
            estimated_tokens=used,
            detail=f"context estimate {used} exceeds budget {token_budget}",
        )

    return ArmResult(
        task_id=task_id,
        adapter=name,
        status=ArmStatus.OK,
        context=context,
        estimated_tokens=used,
    )


def prepare_all(
    adapters: Iterable[ContextAdapter],
    task: dict[str, Any],
    token_budget: int,
) -> list[ArmResult]:
    return [prepare_arm(adapter, task, token_budget) for adapter in adapters]


def comparison_is_claimable(results: Iterable[ArmResult]) -> bool:
    """A comparison is claimable only when every participating arm is valid."""
    rows = list(results)
    return bool(rows) and all(row.valid_measurement for row in rows)
