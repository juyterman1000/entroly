"""Entroly ContextEngine — Hermes Agent plugin implementation.

Maps the four required Hermes ``ContextEngine`` abstract methods to Entroly's
SDK, giving Hermes users query-conditioned selection, evidence receipts, and
recoverable compression instead of lossy summarization.

Design rules:
    - **No Hermes import at module level.**  The ABC is inherited when
      ``agent.context_engine`` is importable; otherwise a compatible base
      is synthesized so ``pip install entroly`` never pulls hermes-agent.
    - **Fail-open.**  Any Entroly error returns messages unchanged.
    - **System prompt protection.**  Hermes's ``<tools>`` / ChatML system
      messages are extracted before compression and prepended after.
    - **Value tracking.**  Savings are recorded through Entroly's existing
      ``value_tracker`` so ``entroly value`` reflects Hermes usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("entroly.hermes")

# ── Inherit from Hermes ABC if available, else synthesize a base ──────────
try:
    from agent.context_engine import ContextEngine as _HermesBase
except ImportError:
    # Outside a Hermes install — create a compatible protocol stub so the
    # class remains instantiable for testing and direct SDK use.
    class _HermesBase:  # type: ignore[no-redef]
        """Stub matching Hermes ContextEngine state fields."""

        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 0
        context_length: int = 0
        compression_count: int = 0
        threshold_percent: float = 0.75
        protect_first_n: int = 3
        protect_last_n: int = 6


class EntrolyContextEngine(_HermesBase):
    """Hermes-compatible context engine backed by Entroly's SDK.

    Replaces Hermes's built-in lossy summarization with:
    - Query-conditioned evidence selection (``focus_topic`` → QCCR ranking)
    - Recoverable compression with receipts
    - Hermes system-prompt and ChatML marker protection
    - Token savings tracking via ``entroly value``
    """

    def __init__(
        self,
        *,
        context_length: int = 200_000,
        threshold_percent: float = 0.75,
        protect_first_n: int = 3,
        protect_last_n: int = 6,
        profile: str = "balanced",
    ) -> None:
        """Initialize the engine.

        Args:
            context_length: Model context window size in tokens.
            threshold_percent: Fire compression when prompt exceeds this
                fraction of ``context_length``.
            protect_first_n: Non-system head messages to keep verbatim.
            protect_last_n: Tail messages to keep verbatim.
            profile: Entroly compression profile (``safe``, ``balanced``,
                ``max``).
        """
        # State fields Hermes run_agent.py reads directly.
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.context_length: int = context_length
        self.threshold_tokens: int = int(context_length * threshold_percent)
        self.threshold_percent: float = threshold_percent
        self.protect_first_n: int = protect_first_n
        self.protect_last_n: int = protect_last_n
        self.compression_count: int = 0

        # Entroly-specific config.
        self._profile: str = profile
        self._session_id: str | None = None
        self._total_tokens_saved: int = 0

    # ── Identity ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Short identifier matching ``context.engine`` in config.yaml."""
        return "entroly"

    # ── Token state tracking ──────────────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response.

        Called after every LLM call.  Hermes normalizes usage into both
        legacy keys (``prompt_tokens``, ``completion_tokens``) and newer
        canonical keys (``input_tokens``, ``output_tokens``).  We accept
        either form.
        """
        self.last_prompt_tokens = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        self.last_completion_tokens = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        self.last_total_tokens = int(
            usage.get("total_tokens")
            or (self.last_prompt_tokens + self.last_completion_tokens)
        )

    # ── Compression trigger ───────────────────────────────────────────────

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        """Return True when the conversation should be compacted.

        Uses the same threshold logic as Hermes's built-in compressor:
        fire when prompt tokens exceed ``threshold_percent`` of the model's
        context window.
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens > self.threshold_tokens

    # ── Core compression ──────────────────────────────────────────────────

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Compact the message list using Entroly's SDK.

        This is the core method Hermes calls when ``should_compress()``
        returns True.

        Args:
            messages: OpenAI-format message list.
            current_tokens: Current estimated token count (from Hermes).
            focus_topic: The user's current task/topic — maps directly to
                Entroly's evidence-selection query for QCCR ranking.

        Returns:
            Compressed message list in OpenAI format.
        """
        if not messages:
            return messages

        try:
            return self._compress_with_entroly(messages, current_tokens, focus_topic)
        except Exception as exc:
            # Fail-open: Entroly errors must never break Hermes.
            logger.warning("Entroly compression failed, returning messages unchanged: %s", exc)
            return messages

    def _compress_with_entroly(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None,
        focus_topic: str | None,
    ) -> List[Dict[str, Any]]:
        """Internal compression — may raise on Entroly errors."""
        from ...sdk import compress_messages

        # ── Protect Hermes system prompts ─────────────────────────────
        # Hermes uses <tools></tools> XML tags and specific system prompts
        # for function calling.  These MUST survive compression verbatim.
        protected_system: List[Dict[str, Any]] = []
        compressible: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if "<tools>" in content or "function calling AI model" in content:
                    protected_system.append(msg)
                    continue
            compressible.append(msg)

        if not compressible:
            return messages

        # ── Compute budget ────────────────────────────────────────────
        # Leave room for the protected system messages and response.
        protected_tokens = sum(
            len(m.get("content", "")) // 4 for m in protected_system
        )
        response_headroom = max(4096, self.context_length // 5)
        budget = max(
            1024,
            self.context_length - protected_tokens - response_headroom,
        )

        # ── Determine preserve_last_n ─────────────────────────────────
        # Hermes's protect_last_n maps to Entroly's preserve_last_n:
        # keep the tail messages verbatim so the agent's recent context
        # is never summarized away.
        preserve_last = max(2, self.protect_last_n)

        # ── Compress via Entroly SDK ──────────────────────────────────
        before_tokens = sum(
            len(m.get("content", "")) // 4
            for m in compressible
            if isinstance(m.get("content"), str)
        )

        compressed = compress_messages(
            compressible,
            budget=budget,
            preserve_last_n=preserve_last,
            profile=self._profile,
        )

        after_tokens = sum(
            len(m.get("content", "")) // 4
            for m in compressed
            if isinstance(m.get("content"), str)
        )

        # ── Track savings ─────────────────────────────────────────────
        saved = max(0, before_tokens - after_tokens)
        self._total_tokens_saved += saved
        self.compression_count += 1

        self._track_value(saved)

        logger.info(
            "Entroly compression #%d: %d → %d tokens (saved %d, budget %d)",
            self.compression_count,
            before_tokens,
            after_tokens,
            saved,
            budget,
        )

        # ── Reassemble: protected system first, then compressed ───────
        return protected_system + compressed

    # ── Lifecycle hooks (optional) ────────────────────────────────────────

    def on_session_start(self, session_id: str = "", **kwargs: Any) -> None:
        """Called when a Hermes conversation begins."""
        self._session_id = session_id
        self._total_tokens_saved = 0
        self.compression_count = 0
        logger.debug("Entroly context engine: session started (%s)", session_id)

    def on_session_end(self, session_id: str = "", messages: list | None = None) -> None:
        """Called at real session boundaries (CLI exit, /reset, gateway expiry)."""
        if self._total_tokens_saved > 0:
            logger.info(
                "Entroly session summary: %d compressions, %d tokens saved",
                self.compression_count,
                self._total_tokens_saved,
            )
        self._session_id = None

    def on_session_reset(self) -> None:
        """Reset internal counters."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self._total_tokens_saved = 0

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _track_value(tokens_saved: int) -> None:
        """Record savings to Entroly's value tracker (best-effort)."""
        if tokens_saved <= 0:
            return
        try:
            from ...value_tracker import get_tracker

            tracker = get_tracker()
            tracker.record(
                tokens_saved=tokens_saved,
                model="",
                optimized=True,
                source="hermes_context_engine",
            )
            tracker.record_event(
                "compress",
                f"Hermes ContextEngine: saved {tokens_saved:,} tokens",
                source="hermes_context_engine",
                tokens_saved=tokens_saved,
            )
        except Exception:
            pass  # Value tracking is best-effort only.
