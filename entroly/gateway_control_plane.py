"""Composable control plane for cache-aware gateway execution."""

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


@dataclass(frozen=True, slots=True)
class GatewayExecutionPlan:
    conversation_id: str
    request: CanonicalGatewayRequest
    stable_prompt: StablePrompt
    failover: FailoverPlan
    routing: CacheRoutingDecision
    redaction: RedactionReceipt


class GatewayControlPlane:
    """Coordinate redaction, capabilities, cache routing, and accounting.

    The class performs no network I/O. A transport executes the failover targets
    in order, then feeds provider-reported usage into observe_response.
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

    def plan(
        self,
        request: CanonicalGatewayRequest,
        *,
        stable_prompt: StablePrompt,
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
        redacted_request, request_receipt = self.redaction_policy.apply(request)
        redacted_prefix, prefix_receipt = self.redaction_policy.redact_text(
            stable_prompt.stable_prefix
        )
        redacted_tail, tail_receipt = self.redaction_policy.redact_text(
            stable_prompt.dynamic_tail
        )
        prompt_findings = prefix_receipt.findings + tail_receipt.findings
        if redacted_prefix != stable_prompt.stable_prefix or redacted_tail != stable_prompt.dynamic_tail:
            stable_prompt = StablePrompt(
                stable_prefix=redacted_prefix,
                dynamic_tail=redacted_tail,
                prefix_hash=hashlib.sha256(redacted_prefix.encode("utf-8")).hexdigest(),
                version=stable_prompt.version,
                section_names=stable_prompt.section_names,
            )
        redaction_receipt = RedactionReceipt(
            enabled=request_receipt.enabled or prefix_receipt.enabled or tail_receipt.enabled,
            changed=request_receipt.changed or prefix_receipt.changed or tail_receipt.changed,
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

        failover = self.failover_planner.plan(
            redacted_request,
            target_choices,
            preferred_key=f"{current.provider}:{current.model}",
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
            provider_failed=provider_failed,
            now=now,
        )
        failed_provider_keys = (
            {
                target.key
                for target in target_choices
                if provider_failed and target.provider == current.provider
            }
        )
        failover = self.failover_planner.plan(
            redacted_request,
            target_choices,
            preferred_key=f"{routing.selected_provider}:{routing.selected_model}",
            excluded_keys=failed_provider_keys,
        )
        if failover.primary.key != (
            f"{routing.selected_provider}:{routing.selected_model}"
        ):
            raise RuntimeError("routed model is absent from the executable failover plan")

        return GatewayExecutionPlan(
            conversation_id=conversation_id,
            request=redacted_request,
            stable_prompt=stable_prompt,
            failover=failover,
            routing=routing,
            redaction=redaction_receipt,
        )

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


__all__ = ["GatewayControlPlane", "GatewayExecutionPlan"]
