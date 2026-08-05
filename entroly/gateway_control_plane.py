"""Composable, same-provider control plane for cache-aware gateway execution."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .cache_routing import CacheAwareRouter, CacheRoutingDecision, ModelCandidate
from .provider_policy import (
    CanonicalGatewayRequest,
    FailoverPlan,
    GatewayRedactionPolicy,
    ProviderFailoverPlanner,
    ProviderTarget,
    RedactionReceipt,
)
from .stable_prefix import StablePrompt, conversation_anchor
from .usage_ledger import (
    UsageEvent,
    UsageLedger,
    UsagePricing,
    parse_provider_usage,
    price_usage,
)


_CROSS_PROVIDER_DISABLED = "cross_provider_disabled"


@dataclass(frozen=True, slots=True)
class GatewayAttemptState:
    """Execution progress used to decide whether an automatic retry is safe.

    Retry is denied by default. A transport must explicitly authorize replay
    after proving that the request is replay-safe and that the previous attempt
    could not have produced an upstream side effect.
    """

    replay_authorized: bool = False
    response_started: bool = False
    tool_call_started: bool = False
    side_effect_possible: bool = True

    def __post_init__(self) -> None:
        values = (
            self.replay_authorized,
            self.response_started,
            self.tool_call_started,
            self.side_effect_possible,
        )
        if any(not isinstance(value, bool) for value in values):
            raise TypeError("gateway attempt state values must be booleans")

    @property
    def automatic_retry_allowed(self) -> bool:
        return self.replay_authorized and not (
            self.response_started
            or self.tool_call_started
            or self.side_effect_possible
        )


@dataclass(frozen=True, slots=True)
class GatewayExecutionPlan:
    conversation_id: str
    source_provider: str
    request: CanonicalGatewayRequest
    stable_prompt: StablePrompt
    failover: FailoverPlan
    routing: CacheRoutingDecision
    redaction: RedactionReceipt

    @property
    def cross_provider_routing(self) -> bool:
        """Return whether an executable target escaped the provider boundary."""
        return any(
            target.provider != self.source_provider
            for target in self.failover.attempts
        )

    def allows_automatic_retry(
        self,
        state: GatewayAttemptState,
        *,
        attempts_completed: int,
    ) -> bool:
        """Allow retry only when an authorized pristine attempt can continue."""
        if (
            isinstance(attempts_completed, bool)
            or not isinstance(attempts_completed, int)
            or attempts_completed < 0
        ):
            raise ValueError("attempts_completed must be a non-negative integer")
        return (
            0 < attempts_completed < len(self.failover.attempts)
            and state.automatic_retry_allowed
        )


class GatewayControlPlane:
    """Coordinate redaction, capabilities, cache routing, and accounting.

    The class performs no network I/O. It intentionally produces executable
    plans containing targets from the original provider only. A transport may
    execute those targets in order and feed provider-reported usage into
    :meth:`observe_response`, but it must also honor
    :class:`GatewayAttemptState` before retrying.

    Cross-provider routing is deliberately outside this control plane. Moving a
    request to another provider changes the credential, data recipient, terms,
    retention, region, and billing boundary and therefore requires a separate,
    explicit operator-authorized feature rather than an automatic fallback.
    """

    def __init__(
        self,
        *,
        cache_router: CacheAwareRouter | None = None,
        failover_planner: ProviderFailoverPlanner | None = None,
        redaction_policy: GatewayRedactionPolicy | None = None,
        usage_ledger: UsageLedger | None = None,
    ) -> None:
        self.cache_router = cache_router or CacheAwareRouter()
        self.failover_planner = failover_planner or ProviderFailoverPlanner()
        self.redaction_policy = redaction_policy or GatewayRedactionPolicy()
        self.usage_ledger = usage_ledger

    def _same_provider_failover(
        self,
        request: CanonicalGatewayRequest,
        targets: Iterable[ProviderTarget],
        *,
        source_provider: str,
        preferred_key: str,
        excluded_keys: Iterable[str] = (),
    ) -> FailoverPlan:
        """Build a failover plan without crossing the provider boundary."""
        target_choices = list(targets)
        target_keys = [target.key for target in target_choices]
        if len(set(target_keys)) != len(target_keys):
            raise ValueError("provider target keys must be unique")

        same_provider = [
            target
            for target in target_choices
            if target.provider == source_provider
        ]
        cross_provider_excluded = {
            target.key: _CROSS_PROVIDER_DISABLED
            for target in target_choices
            if target.provider != source_provider
        }
        if not same_provider:
            raise RuntimeError(
                "no same-provider target is available; "
                "cross-provider routing is disabled"
            )

        plan = self.failover_planner.plan(
            request,
            same_provider,
            preferred_key=preferred_key,
            excluded_keys=excluded_keys,
        )
        return FailoverPlan(
            attempts=plan.attempts,
            excluded={**cross_provider_excluded, **dict(plan.excluded)},
            required_capabilities=plan.required_capabilities,
        )

    def plan(
        self,
        request: CanonicalGatewayRequest,
        *,
        stable_prompt: StablePrompt,
        source_provider: str,
        current_model: str,
        candidates: Iterable[ModelCandidate],
        targets: Iterable[ProviderTarget],
        prefix_tokens: int | None = None,
        new_input_tokens: int = 0,
        expected_output_tokens: int = 0,
        risk: str = "standard",
        expected_turns: int | None = None,
        force_model: str | None = None,
        provider_failed: bool = False,
        now: float | None = None,
    ) -> GatewayExecutionPlan:
        if not isinstance(source_provider, str) or not source_provider.strip():
            raise ValueError("source_provider is required")

        redacted_request, request_receipt = self.redaction_policy.apply(request)
        redacted_prefix, prefix_receipt = self.redaction_policy.redact_text(
            stable_prompt.stable_prefix
        )
        redacted_tail, tail_receipt = self.redaction_policy.redact_text(
            stable_prompt.dynamic_tail
        )
        prompt_findings = prefix_receipt.findings + tail_receipt.findings
        if (
            redacted_prefix != stable_prompt.stable_prefix
            or redacted_tail != stable_prompt.dynamic_tail
        ):
            stable_prompt = StablePrompt(
                stable_prefix=redacted_prefix,
                dynamic_tail=redacted_tail,
                prefix_hash=hashlib.sha256(
                    redacted_prefix.encode("utf-8")
                ).hexdigest(),
                version=stable_prompt.version,
                section_names=stable_prompt.section_names,
            )
        redaction_receipt = RedactionReceipt(
            enabled=(
                request_receipt.enabled
                or prefix_receipt.enabled
                or tail_receipt.enabled
            ),
            changed=(
                request_receipt.changed
                or prefix_receipt.changed
                or tail_receipt.changed
            ),
            findings=request_receipt.findings + prompt_findings,
        )
        choices = list(candidates)
        target_choices = list(targets)
        current_matches = [
            candidate for candidate in choices if candidate.model == current_model
        ]
        if len(current_matches) != 1:
            raise ValueError("current_model must identify exactly one candidate")
        current = current_matches[0]
        if current.provider != source_provider:
            raise ValueError(
                "source_provider does not match the current model candidate"
            )

        if provider_failed:
            raise RuntimeError(
                "source provider failed; cross-provider routing is disabled"
            )

        failover = self._same_provider_failover(
            redacted_request,
            target_choices,
            source_provider=source_provider,
            preferred_key=f"{source_provider}:{current.model}",
        )
        compatible = {target.key for target in failover.attempts}
        routed_candidates = [
            replace(
                candidate,
                capabilities_satisfied=(
                    candidate.capabilities_satisfied
                    and f"{candidate.provider}:{candidate.model}" in compatible
                ),
            )
            for candidate in choices
        ]

        conversation_id = conversation_anchor(
            redacted_request.messages,
            tools=redacted_request.tools,
        )
        routing = self.cache_router.decide(
            conversation_id,
            current_model=current_model,
            candidates=routed_candidates,
            prefix_hash=stable_prompt.prefix_hash,
            prefix_tokens=(
                stable_prompt.stable_tokens_estimate
                if prefix_tokens is None
                else prefix_tokens
            ),
            new_input_tokens=new_input_tokens,
            expected_output_tokens=expected_output_tokens,
            risk=risk,
            expected_turns=expected_turns,
            force_model=force_model,
            provider_failed=False,
            now=now,
        )
        if routing.selected_provider != source_provider:
            raise RuntimeError("cross-provider routing is disabled")

        failover = self._same_provider_failover(
            redacted_request,
            target_choices,
            source_provider=source_provider,
            preferred_key=(
                f"{routing.selected_provider}:{routing.selected_model}"
            ),
        )
        if failover.primary.key != (
            f"{routing.selected_provider}:{routing.selected_model}"
        ):
            raise RuntimeError(
                "routed model is absent from the executable failover plan"
            )

        plan = GatewayExecutionPlan(
            conversation_id=conversation_id,
            source_provider=source_provider,
            request=redacted_request,
            stable_prompt=stable_prompt,
            failover=failover,
            routing=routing,
            redaction=redaction_receipt,
        )
        if plan.cross_provider_routing:
            raise RuntimeError("cross-provider executable target escaped policy")
        return plan

    def observe_response(
        self,
        plan: GatewayExecutionPlan,
        *,
        request_id: str,
        provider: str,
        model: str,
        usage_payload: Mapping[str, Any],
        pricing: UsagePricing,
        observed_at: float | None = None,
        cache_ttl_seconds: float | None = None,
        team: str = "",
        tool: str = "",
        project: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> UsageEvent:
        if provider != plan.source_provider:
            raise ValueError(
                "response provider does not match the plan source provider"
            )
        observed_key = f"{provider}:{model}"
        executable_keys = {target.key for target in plan.failover.attempts}
        if observed_key not in executable_keys:
            raise ValueError("response target was not present in the execution plan")

        usage = parse_provider_usage(provider, usage_payload)
        timestamp = time.time() if observed_at is None else observed_at
        self.cache_router.observe(
            plan.conversation_id,
            model=model,
            provider=provider,
            prefix_hash=plan.stable_prompt.prefix_hash,
            cached_prefix_tokens=max(
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            ),
            cache_hit=usage.cache_read_tokens > 0,
            observed_at=timestamp,
            ttl_seconds=cache_ttl_seconds,
        )

        if self.usage_ledger is None:
            cost, savings = price_usage(usage, pricing)
            return UsageEvent(
                request_id=request_id,
                provider=provider,
                model=model,
                usage=usage,
                cost_micro_usd=cost,
                cache_savings_micro_usd=savings,
                occurred_at=timestamp,
                team=team,
                tool=tool,
                project=project,
                conversation_id=plan.conversation_id,
                pricing_source=pricing.source,
                metadata=dict(metadata or {}),
            )

        return self.usage_ledger.record_provider_payload(
            request_id=request_id,
            provider=provider,
            model=model,
            payload=usage_payload,
            pricing=pricing,
            occurred_at=timestamp,
            team=team,
            tool=tool,
            project=project,
            conversation_id=plan.conversation_id,
            metadata=metadata,
        )


__all__ = [
    "GatewayAttemptState",
    "GatewayControlPlane",
    "GatewayExecutionPlan",
]
