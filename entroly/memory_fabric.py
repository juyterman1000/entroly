"""Entroly Memory Fabric — one public orchestration layer for agent memory.

MemoryOS is the stable local runtime. The Fabric wraps it with capability
introspection and safe extension points for the deeper memory ecosystem:

- built-in MemoryOS for local safety, budget-aware recall, receipts, save/load,
- production Python fallback kernels for IPC, compliance, and pollination,
- optional hippocampus-sharp-memory bridge for cross-session long-term memory,
- optional native Rust kernels when the installed entroly_core wheel exports them,
- future Federation hooks for privacy-preserving shared learning.

The key product contract is that applications can depend on this class today
without caring which optional memory engines are installed. Capabilities are
reported explicitly instead of being hidden behind import side effects.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .memory import MemoryContext, MemoryOS, MemoryTier, SafetyPolicy
from .memory_kernels import ComplianceKernel, PollinationKernel, SchipcBus

MemoryLayerStatus = Literal[
    "active",
    "available",
    "optional",
    "internal",
    "missing",
    "disabled",
]


@dataclass(slots=True)
class MemoryLayer:
    """Capability record for one layer of the Entroly memory stack."""

    name: str
    status: MemoryLayerStatus
    role: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class FabricRecall:
    """Unified recall result returned by MemoryFabric.recall()."""

    context: MemoryContext
    long_term: list[dict[str, object]] = field(default_factory=list)
    layers: list[MemoryLayer] = field(default_factory=list)

    def as_text(self) -> str:
        blocks: list[str] = []
        if self.context.as_text():
            blocks.append(self.context.as_text())
        if self.long_term:
            rendered = []
            for idx, mem in enumerate(self.long_term, start=1):
                source = mem.get("source", "long_term_memory")
                retention = mem.get("retention", "?")
                content = mem.get("content", "")
                rendered.append(
                    f"[long-term:{idx} source={source} "
                    f"retention={retention}]\n{content}"
                )
            blocks.append("\n\n".join(rendered))
        return "\n\n".join(blocks)

    def receipt(self) -> dict[str, object]:
        return {
            "memory_os": self.context.receipt(),
            "long_term": self.long_term,
            "layers": [layer.as_dict() for layer in self.layers],
        }


class MemoryFabric:
    """Unified public facade over Entroly's memory ecosystem.

    Use this when an application wants the best available memory stack without
    directly depending on optional packages or native bindings.
    """

    def __init__(
        self,
        *,
        default_budget: int = 4096,
        max_entries: int = 50_000,
        max_tokens: int = 500_000,
        safety_policy: SafetyPolicy = "block",
        enable_long_term: bool = True,
        enable_native: bool = True,
        enable_builtin_kernels: bool = True,
    ) -> None:
        self.memory_os = MemoryOS(
            default_budget=default_budget,
            max_entries=max_entries,
            max_tokens=max_tokens,
            safety_policy=safety_policy,
        )
        self._long_term = None
        self._native_memory_cls = None
        self._native_memory = None
        self._ipc_kernel = None
        self._compliance_kernel = None
        self._pollination_kernel = None
        self._kernel_origins: dict[str, str] = {}
        self._long_term_error: str | None = None
        self._native_error: str | None = None
        self._native_errors: dict[str, str] = {}
        self._enable_long_term = enable_long_term
        self._enable_native = enable_native
        self._enable_builtin_kernels = enable_builtin_kernels

        if enable_long_term:
            self._init_long_term()
        if enable_native:
            self._init_native_kernels()
        if enable_builtin_kernels:
            self._init_builtin_kernels()

    @classmethod
    def load(cls, path: str | os.PathLike[str], **kwargs: object) -> "MemoryFabric":
        fabric = cls(**kwargs)
        fabric.memory_os = MemoryOS.load(path)
        return fabric

    def save(self, path: str | os.PathLike[str]) -> Path:
        return self.memory_os.save(path)

    def remember(
        self,
        content: str,
        *,
        agent_id: str = "default",
        importance: float = 0.5,
        tier: MemoryTier = "working",
        source: str = "manual",
        tags: Iterable[str] | None = None,
        safety_policy: SafetyPolicy | None = None,
    ) -> str:
        """Store memory in the stable local MemoryOS layer."""
        return self.memory_os.remember(
            content,
            agent_id=agent_id,
            importance=importance,
            tier=tier,
            source=source,
            tags=tags,
            safety_policy=safety_policy,
        )

    def recall(
        self,
        query: str = "",
        *,
        agent_id: str = "default",
        budget: int | None = None,
        tier: MemoryTier | None = None,
        include_shared: bool = True,
        long_term_top_k: int = 5,
    ) -> FabricRecall:
        """Recall from MemoryOS and, when available, optional long-term memory."""
        context = self.memory_os.recall(
            query,
            agent_id=agent_id,
            budget=budget,
            tier=tier,
            include_shared=include_shared,
        )
        long_term: list[dict[str, object]] = []
        if self._long_term is not None and getattr(self._long_term, "active", False) and query:
            try:
                long_term = list(
                    self._long_term.recall_relevant(query, top_k=long_term_top_k)
                )
            except Exception as exc:  # pragma: no cover
                self._long_term_error = str(exc)
                long_term = []
        return FabricRecall(context=context, long_term=long_term, layers=self.capabilities())

    def send_agent_message(
        self,
        sender_id: int,
        receiver_id: int,
        content: str,
    ) -> dict[str, object]:
        """Send a multi-agent memory/message through safety + SCHIPC filtering.

        Uses native kernels when exported by entroly_core, otherwise uses the
        dependency-free Python production fallback kernels.
        """
        if self._compliance_kernel is not None:
            decision = self._compliance_kernel.check_message(sender_id, receiver_id, content)
            decision_dict = _pyobject_to_dict(decision)
            if decision_dict.get("allowed") is False:
                return {
                    "delivered": False,
                    "reason": decision_dict.get("reason", "blocked"),
                    "compliance": decision_dict,
                    "ipc": None,
                    "kernel_origin": self._kernel_origins.get("compliance_gate", "unknown"),
                }
        else:
            return {
                "delivered": False,
                "reason": "compliance_gate_unavailable",
                "compliance": {"allowed": None, "reason": "compliance_gate_unavailable"},
                "ipc": None,
                "kernel_origin": "none",
            }

        if self._ipc_kernel is None:
            return {
                "delivered": False,
                "reason": "ipc_unavailable",
                "compliance": decision_dict,
                "ipc": None,
                "kernel_origin": "none",
            }

        ipc_result = _pyobject_to_dict(self._ipc_kernel.send(sender_id, receiver_id, content))
        return {
            "delivered": bool(ipc_result.get("delivered")),
            "reason": ipc_result.get("reason", "unknown"),
            "compliance": decision_dict,
            "ipc": ipc_result,
            "kernel_origin": self._kernel_origins.get("schipc", "unknown"),
        }

    def record_agent_lesson(
        self,
        agent_id: str,
        description: str,
        *,
        success: bool = True,
        surprise: float = 0.0,
        domain: str = "general",
    ) -> dict[str, object]:
        """Record a lesson in the pollination kernel."""
        if self._pollination_kernel is None:
            return {"recorded": False, "reason": "pollination_unavailable"}
        self._pollination_kernel.register_agent(agent_id)
        self._pollination_kernel.record_lesson(agent_id, description, success, surprise, domain)
        return {
            "recorded": True,
            "agent_id": agent_id,
            "kernel_origin": self._kernel_origins.get("pollination", "unknown"),
        }

    def share_agent_lessons(self, from_agent: str, to_agent: str) -> dict[str, object]:
        """Share learned lessons through the pollination kernel."""
        if self._pollination_kernel is None:
            return {"shared": False, "reason": "pollination_unavailable", "count": 0}
        self._pollination_kernel.register_agent(from_agent)
        self._pollination_kernel.register_agent(to_agent)
        count = int(self._pollination_kernel.share(from_agent, to_agent))
        return {
            "shared": count > 0,
            "count": count,
            "kernel_origin": self._kernel_origins.get("pollination", "unknown"),
        }

    def reward_agent_share(self, sharer_id: str, receiver_id: str, reward: float) -> None:
        """Reward or penalize a pollination share when available."""
        if self._pollination_kernel is not None:
            self._pollination_kernel.reward(sharer_id, receiver_id, reward)

    def tick(self, count: int = 1) -> None:
        self.memory_os.tick(count)
        if self._long_term is not None:
            for _ in range(max(1, int(count))):
                try:
                    self._long_term.tick()
                except Exception:  # pragma: no cover
                    break
        if self._pollination_kernel is not None:
            for _ in range(max(1, int(count))):
                try:
                    self._pollination_kernel.tick()
                except Exception:  # pragma: no cover
                    break

    def forget(self, threshold: float | None = None) -> int:
        return self.memory_os.forget(threshold)

    def consolidate(self) -> dict[str, object]:
        result: dict[str, object] = {"memory_os_promoted": self.memory_os.consolidate()}
        if self._long_term is not None and getattr(self._long_term, "active", False):
            try:
                result["long_term"] = self._long_term.consolidate()
            except Exception as exc:  # pragma: no cover
                result["long_term_error"] = str(exc)
        return result

    def stats(self) -> dict[str, object]:
        stats: dict[str, object] = {
            "memory_os": self.memory_os.stats(),
            "layers": [layer.as_dict() for layer in self.capabilities()],
            "kernel_origins": dict(self._kernel_origins),
        }
        if self._long_term is not None:
            try:
                stats["long_term"] = self._long_term.stats()
            except Exception as exc:  # pragma: no cover
                stats["long_term"] = {"active": False, "error": str(exc)}
        kernels: dict[str, object] = {}
        if self._native_memory is not None and hasattr(self._native_memory, "stats"):
            kernels["memory_manager"] = _pyobject_to_dict(self._native_memory.stats())
        if self._ipc_kernel is not None:
            kernels["ipc"] = _pyobject_to_dict(self._ipc_kernel.stats())
        if self._compliance_kernel is not None:
            kernels["compliance"] = _pyobject_to_dict(self._compliance_kernel.stats())
        if self._pollination_kernel is not None:
            kernels["pollination"] = _pyobject_to_dict(self._pollination_kernel.stats())
        if kernels:
            stats["kernels"] = kernels
        return stats

    def capabilities(self) -> list[MemoryLayer]:
        layers = [
            MemoryLayer(
                "memory_os",
                "active",
                "public runtime memory",
                "Local safety, budget-aware recall, decay, persistence, and receipts.",
            ),
        ]

        if self._long_term is not None and getattr(self._long_term, "active", False):
            layers.append(
                MemoryLayer(
                    "hippocampus_bridge",
                    "active",
                    "optional long-term memory",
                    "hippocampus-sharp-memory is installed and active.",
                )
            )
        elif self._enable_long_term:
            detail = (
                self._long_term_error
                or "Install hippocampus-sharp-memory to activate cross-session memory."
            )
            layers.append(
                MemoryLayer("hippocampus_bridge", "optional", "optional long-term memory", detail)
            )
        else:
            layers.append(
                MemoryLayer(
                    "hippocampus_bridge",
                    "disabled",
                    "optional long-term memory",
                    "Disabled by configuration.",
                )
            )

        layers.append(
            self._kernel_layer(
                "rust_memory_manager",
                self._native_memory,
                self._kernel_origins.get("rust_memory_manager"),
                "native high-scale memory",
                "entroly_core.MemoryManager was detected.",
                "Native MemoryManager not exported by installed entroly_core yet.",
            )
        )
        layers.append(
            self._kernel_layer(
                "schipc",
                self._ipc_kernel,
                self._kernel_origins.get("schipc"),
                "multi-agent memory traffic",
                "SCHIPC kernel is active.",
                "SCHIPC unavailable; enable built-in kernels or native exports.",
            )
        )
        layers.append(
            self._kernel_layer(
                "compliance_gate",
                self._compliance_kernel,
                self._kernel_origins.get("compliance_gate"),
                "memory safety kernel",
                "Compliance kernel is active.",
                "ComplianceGate unavailable; enable built-in kernels or native exports.",
            )
        )
        layers.append(
            self._kernel_layer(
                "pollination",
                self._pollination_kernel,
                self._kernel_origins.get("pollination"),
                "learned agent lesson sharing",
                "Pollination kernel is active.",
                "Pollination unavailable; enable built-in kernels or native exports.",
            )
        )
        layers.extend(
            [
                MemoryLayer(
                    "federation",
                    "optional",
                    "privacy-preserving shared learning",
                    "FederationClient shares noised archetype weights when enabled.",
                ),
                MemoryLayer(
                    "receipts_witness",
                    "active",
                    "trust and audit layer",
                    "Memory recall can pair with Context Receipts and WITNESS.",
                ),
            ]
        )
        return layers

    def _kernel_layer(
        self,
        name: str,
        instance: object | None,
        origin: str | None,
        role: str,
        available_detail: str,
        missing_detail: str,
    ) -> MemoryLayer:
        if instance is not None:
            detail = available_detail if origin == "native" else f"Python fallback active: {available_detail}"
            return MemoryLayer(name, "available", role, detail)
        if not self._enable_native and not self._enable_builtin_kernels:
            return MemoryLayer(name, "disabled", role, "Disabled by configuration.")
        return MemoryLayer(name, "internal", role, self._native_errors.get(name, missing_detail))

    def _init_long_term(self) -> None:
        try:
            from .long_term_memory import LongTermMemory  # noqa: PLC0415

            self._long_term = LongTermMemory()
        except Exception as exc:  # pragma: no cover
            self._long_term_error = str(exc)
            self._long_term = None

    def _init_native_kernels(self) -> None:
        try:
            import entroly_core  # type: ignore[import-not-found]  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            self._native_error = str(exc)
            self._native_errors["rust_memory_manager"] = str(exc)
            self._native_errors["schipc"] = str(exc)
            self._native_errors["compliance_gate"] = str(exc)
            self._native_errors["pollination"] = str(exc)
            return

        self._native_memory = self._instantiate_native(
            entroly_core,
            "MemoryManager",
            "rust_memory_manager",
        )
        if self._native_memory is not None:
            self._kernel_origins["rust_memory_manager"] = "native"
        self._native_memory_cls = getattr(entroly_core, "MemoryManager", None)
        self._ipc_kernel = self._instantiate_native(entroly_core, "IpcBus", "schipc")
        if self._ipc_kernel is not None:
            self._kernel_origins["schipc"] = "native"
        self._compliance_kernel = self._instantiate_native(
            entroly_core,
            "ComplianceGate",
            "compliance_gate",
        )
        if self._compliance_kernel is not None:
            self._kernel_origins["compliance_gate"] = "native"
        self._pollination_kernel = self._instantiate_native(
            entroly_core,
            "PollinationEngine",
            "pollination",
        )
        if self._pollination_kernel is not None:
            self._kernel_origins["pollination"] = "native"

    def _init_builtin_kernels(self) -> None:
        if self._ipc_kernel is None:
            self._ipc_kernel = SchipcBus()
            self._kernel_origins["schipc"] = "python_fallback"
        if self._compliance_kernel is None:
            self._compliance_kernel = ComplianceKernel()
            self._kernel_origins["compliance_gate"] = "python_fallback"
        if self._pollination_kernel is None:
            self._pollination_kernel = PollinationKernel()
            self._kernel_origins["pollination"] = "python_fallback"

    def _instantiate_native(self, module: object, attr: str, layer_name: str) -> object | None:
        cls = getattr(module, attr, None)
        if cls is None:
            self._native_errors[layer_name] = f"entroly_core.{attr} is not exported"
            return None
        try:
            return cls()
        except Exception as exc:  # pragma: no cover
            self._native_errors[layer_name] = str(exc)
            return None


def _pyobject_to_dict(value: Any) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return {"value": value}


__all__ = ["FabricRecall", "MemoryFabric", "MemoryLayer", "MemoryLayerStatus"]
