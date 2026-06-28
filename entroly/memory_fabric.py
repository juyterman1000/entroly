"""Entroly Memory Fabric — one public orchestration layer for agent memory.

MemoryOS is the stable local runtime. The fabric wraps it with capability
introspection and safe extension points for the deeper memory ecosystem:

- built-in MemoryOS for local safety, budget-aware recall, receipts, save/load,
- optional hippocampus-sharp-memory bridge for cross-session long-term memory,
- optional native Rust MemoryManager for future high-scale recall,
- optional native SCHIPC / ComplianceGate / Pollination kernels when exported,
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
        self._native_ipc = None
        self._native_compliance = None
        self._native_pollination = None
        self._long_term_error: str | None = None
        self._native_error: str | None = None
        self._native_errors: dict[str, str] = {}
        self._enable_long_term = enable_long_term
        self._enable_native = enable_native

        if enable_long_term:
            self._init_long_term()
        if enable_native:
            self._init_native_kernels()

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
        """Store memory in the stable local MemoryOS layer.

        Native MemoryManager remains optional because the public contract must
        stay deterministic and safe even when native wheels are absent. When the
        installed native package exports MemoryManager, the Fabric detects it and
        reports it through capabilities/stats so applications can opt into it.
        """
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
        """Send a multi-agent memory/message through native safety + SCHIPC when available.

        If native kernels are absent, this method fails closed with a structured
        disabled result instead of silently pretending that IPC filtering ran.
        """
        if self._native_compliance is not None:
            decision = self._native_compliance.check_message(sender_id, receiver_id, content)
            decision_dict = _pyobject_to_dict(decision)
            if decision_dict.get("allowed") is False:
                return {
                    "delivered": False,
                    "reason": decision_dict.get("reason", "blocked"),
                    "compliance": decision_dict,
                    "ipc": None,
                }
        else:
            decision_dict = {"allowed": None, "reason": "compliance_gate_unavailable"}

        if self._native_ipc is None:
            return {
                "delivered": False,
                "reason": "native_ipc_unavailable",
                "compliance": decision_dict,
                "ipc": None,
            }

        ipc_result = _pyobject_to_dict(self._native_ipc.send(sender_id, receiver_id, content))
        return {
            "delivered": bool(ipc_result.get("delivered")),
            "reason": ipc_result.get("reason", "unknown"),
            "compliance": decision_dict,
            "ipc": ipc_result,
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
        """Record a lesson in native PollinationEngine when available."""
        if self._native_pollination is None:
            return {"recorded": False, "reason": "native_pollination_unavailable"}
        self._native_pollination.register_agent(agent_id)
        self._native_pollination.record_lesson(agent_id, description, success, surprise, domain)
        return {"recorded": True, "agent_id": agent_id}

    def share_agent_lessons(self, from_agent: str, to_agent: str) -> dict[str, object]:
        """Share learned lessons through native PollinationEngine when available."""
        if self._native_pollination is None:
            return {"shared": False, "reason": "native_pollination_unavailable", "count": 0}
        self._native_pollination.register_agent(from_agent)
        self._native_pollination.register_agent(to_agent)
        count = int(self._native_pollination.share(from_agent, to_agent))
        return {"shared": count > 0, "count": count}

    def reward_agent_share(self, sharer_id: str, receiver_id: str, reward: float) -> None:
        """Reward or penalize a native pollination share when available."""
        if self._native_pollination is not None:
            self._native_pollination.reward(sharer_id, receiver_id, reward)

    def tick(self, count: int = 1) -> None:
        self.memory_os.tick(count)
        if self._long_term is not None:
            for _ in range(max(1, int(count))):
                try:
                    self._long_term.tick()
                except Exception:  # pragma: no cover
                    break
        if self._native_pollination is not None:
            for _ in range(max(1, int(count))):
                try:
                    self._native_pollination.tick()
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
        }
        if self._long_term is not None:
            try:
                stats["long_term"] = self._long_term.stats()
            except Exception as exc:  # pragma: no cover
                stats["long_term"] = {"active": False, "error": str(exc)}
        native: dict[str, object] = {}
        if self._native_memory is not None and hasattr(self._native_memory, "stats"):
            native["memory_manager"] = _pyobject_to_dict(self._native_memory.stats())
        if self._native_ipc is not None:
            native["ipc"] = _pyobject_to_dict(self._native_ipc.stats())
        if self._native_compliance is not None:
            native["compliance"] = _pyobject_to_dict(self._native_compliance.stats())
        if self._native_pollination is not None:
            native["pollination"] = _pyobject_to_dict(self._native_pollination.stats())
        if native:
            stats["native"] = native
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
            self._native_layer(
                "rust_memory_manager",
                self._native_memory,
                "native high-scale memory",
                "entroly_core.MemoryManager was detected.",
                "Native MemoryManager not exported by installed entroly_core yet.",
            )
        )
        layers.append(
            self._native_layer(
                "schipc",
                self._native_ipc,
                "multi-agent memory traffic",
                "entroly_core.IpcBus was detected and instantiated.",
                "Rust IPC bus exists in source; PyO3 export pending in installed wheel.",
            )
        )
        layers.append(
            self._native_layer(
                "compliance_gate",
                self._native_compliance,
                "memory safety kernel",
                "entroly_core.ComplianceGate was detected and instantiated.",
                "Rust compliance gate exists in source; PyO3 export pending in installed wheel.",
            )
        )
        layers.append(
            self._native_layer(
                "pollination",
                self._native_pollination,
                "learned agent lesson sharing",
                "entroly_core.PollinationEngine was detected and instantiated.",
                "Rust TD(0) pollination exists in source; PyO3 export pending in installed wheel.",
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

    def _native_layer(
        self,
        name: str,
        instance: object | None,
        role: str,
        available_detail: str,
        missing_detail: str,
    ) -> MemoryLayer:
        if instance is not None:
            return MemoryLayer(name, "available", role, available_detail)
        if not self._enable_native:
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
        self._native_memory_cls = getattr(entroly_core, "MemoryManager", None)
        self._native_ipc = self._instantiate_native(entroly_core, "IpcBus", "schipc")
        self._native_compliance = self._instantiate_native(
            entroly_core,
            "ComplianceGate",
            "compliance_gate",
        )
        self._native_pollination = self._instantiate_native(
            entroly_core,
            "PollinationEngine",
            "pollination",
        )

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
