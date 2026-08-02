"""Executable, offline provider protocol conformance checks.

The conformance report is deliberately conservative. It verifies adapter
behavior without provider credentials or network requests and distinguishes
same-provider preservation from cross-provider semantic translation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .provider_adapters import (
    apply_target_same_provider,
    canonical_request_from_provider_body,
    render_canonical_request,
)
from .provider_policy import Capability, ProviderTarget

SCHEMA_VERSION = "entroly.provider-conformance.v1"
_FULL_CAPABILITIES = frozenset(Capability)


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    name: str
    source_provider: str
    target_provider: str
    body: Mapping[str, Any]
    expected_allowed: bool
    path: str = ""


def provider_capability_matrix() -> dict[str, Any]:
    """Return the implemented protocol contract without connectivity claims."""
    protocols = {
        "openai_chat_completions": "openai",
        "openai_responses": "responses",
        "anthropic_messages": "anthropic",
        "gemini_generate_content": "gemini",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocols": {
            name: {
                "adapter": provider,
                "same_provider": {
                    "opaque_fields_preserved": True,
                    "tools_preserved": True,
                    "schemas_preserved": True,
                    "vision_preserved": True,
                    "reasoning_preserved": True,
                    "cache_control_preserved": True,
                    "streaming_preserved": True,
                },
                "cross_provider": {
                    "text_only_messages": True,
                    "tools": False,
                    "schemas": False,
                    "vision": False,
                    "reasoning": False,
                    "cache_control": False,
                    "unmapped_fields": False,
                },
                "connectivity_verified": False,
            }
            for name, provider in protocols.items()
        },
        "claims": {
            "provider_connectivity_verified": False,
            "full_semantic_equivalence_implied": False,
        },
    }


def _target(provider: str) -> ProviderTarget:
    return ProviderTarget(
        provider=provider,
        model="conformance-model",
        capabilities=_FULL_CAPABILITIES,
    )


def assess_provider_request(
    source_provider: str,
    target_provider: str,
    body: Mapping[str, Any],
    *,
    path: str = "",
) -> dict[str, Any]:
    """Assess whether a request can be rendered without semantic loss."""
    normalized_source = source_provider.lower()
    normalized_target = target_provider.lower()
    canonicalized = canonical_request_from_provider_body(
        normalized_source,
        body,
        path=path,
    )
    canonical = canonicalized.canonical
    required = sorted(capability.value for capability in canonical.required_capabilities())

    if normalized_source == "responses":
        same_provider = normalized_target == "openai"
    else:
        same_provider = normalized_source == normalized_target

    if same_provider:
        provider = "openai" if normalized_source == "responses" else normalized_source
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "source-model:generateContent"
            if provider == "gemini"
            else f"https://example.invalid/{provider}"
        )
        rewritten, _ = apply_target_same_provider(
            provider=provider,
            target=_target(provider),
            body=body,
            url=url,
        )
        return {
            "allowed": True,
            "mode": "same_provider_preserve",
            "required_capabilities": required,
            "nonportable_fields": list(canonical.nonportable_fields),
            "preserved_top_level_fields": sorted(str(key) for key in rewritten),
            "reason": None,
        }

    try:
        render_canonical_request(canonical, _target(normalized_target))
    except ValueError as exc:
        return {
            "allowed": False,
            "mode": "cross_provider_fail_closed",
            "required_capabilities": required,
            "nonportable_fields": list(canonical.nonportable_fields),
            "preserved_top_level_fields": [],
            "reason": str(exc),
        }

    return {
        "allowed": True,
        "mode": "cross_provider_text_only",
        "required_capabilities": required,
        "nonportable_fields": [],
        "preserved_top_level_fields": [],
        "reason": None,
    }


def _cases() -> tuple[ConformanceCase, ...]:
    return (
        ConformanceCase(
            "openai_text_to_anthropic",
            "openai",
            "anthropic",
            {"model": "gpt", "messages": [{"role": "user", "content": "hello"}]},
            True,
        ),
        ConformanceCase(
            "anthropic_text_to_openai",
            "anthropic",
            "openai",
            {"model": "claude", "messages": [{"role": "user", "content": "hello"}]},
            True,
        ),
        ConformanceCase(
            "gemini_text_to_openai",
            "gemini",
            "openai",
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            True,
            path="/v1beta/models/gemini-safe:generateContent",
        ),
        ConformanceCase(
            "tools_fail_closed",
            "openai",
            "anthropic",
            {
                "model": "gpt",
                "messages": [{"role": "user", "content": "lookup"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
            False,
        ),
        ConformanceCase(
            "schema_fail_closed",
            "openai",
            "anthropic",
            {
                "model": "gpt",
                "messages": [{"role": "user", "content": "json"}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
            },
            False,
        ),
        ConformanceCase(
            "vision_fail_closed",
            "openai",
            "anthropic",
            {
                "model": "gpt",
                "messages": [{
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}],
                }],
            },
            False,
        ),
        ConformanceCase(
            "reasoning_fail_closed",
            "openai",
            "anthropic",
            {
                "model": "gpt",
                "messages": [{"role": "user", "content": "reason"}],
                "reasoning_effort": "medium",
            },
            False,
        ),
        ConformanceCase(
            "cache_control_fail_closed",
            "anthropic",
            "openai",
            {
                "model": "claude",
                "messages": [{
                    "role": "user",
                    "content": [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}],
                }],
            },
            False,
        ),
        ConformanceCase(
            "unknown_field_fail_closed",
            "openai",
            "anthropic",
            {
                "model": "gpt",
                "messages": [{"role": "user", "content": "hello"}],
                "future_provider_control": {"mode": "strict"},
            },
            False,
        ),
        ConformanceCase(
            "same_provider_preserves_unknown_field",
            "openai",
            "openai",
            {
                "model": "gpt",
                "messages": [{"role": "user", "content": "hello"}],
                "future_provider_control": {"mode": "strict"},
            },
            True,
        ),
    )


def run_provider_conformance() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in _cases():
        assessment = assess_provider_request(
            case.source_provider,
            case.target_provider,
            case.body,
            path=case.path,
        )
        passed = assessment["allowed"] is case.expected_allowed
        results.append({
            "name": case.name,
            "passed": passed,
            "expected_allowed": case.expected_allowed,
            **assessment,
        })

    passed_count = sum(1 for result in results if result["passed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "healthy": passed_count == len(results),
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
        },
        "cases": results,
        "matrix": provider_capability_matrix()["protocols"],
        "claims": {
            "provider_connectivity_verified": False,
            "full_semantic_equivalence_implied": False,
            "production_readiness_implied": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Entroly provider conformance offline")
    parser.add_argument("--json", action="store_true", help="emit stable JSON")
    args = parser.parse_args(argv)
    report = run_provider_conformance()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        summary = report["summary"]
        print(
            f"provider conformance: {summary['passed']}/{summary['total']} passed; "
            "connectivity not checked"
        )
        for case in report["cases"]:
            marker = "PASS" if case["passed"] else "FAIL"
            print(f"  {marker} {case['name']}: {case['mode']}")
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "assess_provider_request",
    "provider_capability_matrix",
    "run_provider_conformance",
]
