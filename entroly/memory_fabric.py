"""Entroly Memory Fabric — one public orchestration layer for agent memory.

MemoryOS is the stable local runtime. The fabric wraps it with capability
introspection and safe extension points for the deeper memory ecosystem:

- built-in MemoryOS for local safety, budget-aware recall, receipts, save/load,
- optional hippocampus-sharp-memory bridge for cross-session long-term memory,
- optional native Rust MemoryManager for future high-scale recall,
- future SCHIPC / ComplianceGate / Pollination / Federation hooks.

The key product contract is that applications can depend on this class today
without caring which optional memory engines are installed. Capabilities are
reported explicitly instead of being hidden behind import side effects.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

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
        self._long_term_error: str | None = None
        self._native_error: str | None = None
        self._enable_long_term = enable_long_term
        self._enable_native = enable_native

        if enable_long_term:
            self._init_long_term()
        if enable_native:
            self._init_native_memory()

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

        Native and long-term layers remain extension points because the public
        contract must stay deterministic and safe even when optional engines are
        absent. Context-selected fragments are still auto-remembered by the
        existing long_term_memory bridge when that bridge is used inside the
        server optimization path.
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

    def tick(self, count: int = 1) -> None:
        self.memory_os.tick(count)
        if self._long_term is not None:
            for _ in range(max(1, int(count))):
                try:
                    self._long_term.tick()
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
        stats = {
            "memory_os": self.memory_os.stats(),
            "layers": [layer.as_dict() for layer in self.capabilities()],
        }
        if self._long_term is not None:
            try:
                stats["long_term"] = self._long_term.stats()
            except Exception as exc:  # pragma: no cover
                stats["long_term"] = {"active": False, "error": str(exc)}
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

        if self._native_memory is not None:
            layers.append(
                MemoryLayer(
                    "rust_memory_manager",
                    "available",
                    "native high-scale memory",
                    "entroly_core.MemoryManager was detected.",
                )
            )
        elif self._enable_native:
            detail = (
                self._native_error
                or "Native MemoryManager not exported by installed entroly_core yet."
            )
            layers.append(
                MemoryLayer("rust_memory_manager", "internal", "native high-scale memory", detail)
            )
        else:
            layers.append(
                MemoryLayer(
                    "rust_memory_manager",
                    "disabled",
                    "native high-scale memory",
                    "Disabled by configuration.",
                )
            )

        layers.extend(
            [
                MemoryLayer(
                    "schipc",
                    "internal",
                    "multi-agent memory traffic",
                    "Rust IPC bus suppresses redundant agent messages; examples pending.",
                ),
                MemoryLayer(
                    "compliance_gate",
                    "internal",
                    "memory safety kernel",
                    "Rust compliance gate exists; MemoryOS exposes safety scan today.",
                ),
                MemoryLayer(
                    "pollination",
                    "internal",
                    "learned agent lesson sharing",
                    "Rust TD(0) pollination engine exists; integration guide pending.",
                ),
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

    def _init_long_term(self) -> None:
        try:
            from .long_term_memory import LongTermMemory  # noqa: PLC0415

            self._long_term = LongTermMemory()
        except Exception as exc:  # pragma: no cover
            self._long_term_error = str(exc)
            self._long_term = None

    def _init_native_memory(self) -> None:
        try:
            from entroly_core import MemoryManager  # type: ignore[import-not-found]  # noqa: PLC0415

            self._native_memory_cls = MemoryManager
            self._native_memory = MemoryManager()
        except Exception as exc:  # pragma: no cover
            self._native_error = str(exc)
            self._native_memory_cls = None
            self._native_memory = None


__all__ = ["FabricRecall", "MemoryFabric", "MemoryLayer", "MemoryLayerStatus"]
