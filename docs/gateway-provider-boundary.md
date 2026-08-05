# Gateway provider boundary

This document defines the safety and claim boundary for
`GatewayControlPlane` and any future integration with `PromptCompilerProxy`.

## Supported execution boundary

`GatewayControlPlane` is a transport-free planner. The caller must pass the
provider detected from the inbound request as `source_provider`; planning fails
when that value does not match the current model candidate.

The control plane may select another model only when that model belongs to the
same source provider. Cross-provider targets are removed from the executable
attempt list and recorded in `FailoverPlan.excluded` with the reason
`cross_provider_disabled`.

Examples:

| Planned transition | Result |
| --- | --- |
| OpenAI model A -> OpenAI model B | Eligible when capabilities and routing policy permit it |
| Anthropic model A -> Anthropic model B | Eligible when capabilities and routing policy permit it |
| Gemini model A -> Gemini model B | Eligible when capabilities and routing policy permit it |
| OpenAI -> Anthropic | Excluded: `cross_provider_disabled` |
| Anthropic -> Gemini | Excluded: `cross_provider_disabled` |
| Gemini -> OpenAI-compatible custom endpoint | Excluded: `cross_provider_disabled` |

Neither lower projected cost nor an explicit `force_model` value can override
this boundary. A source-provider failure also fails closed instead of activating
a different provider. Duplicate provider/model target keys are rejected so a
caller cannot manufacture extra retry capacity for the same billable target.

## Retry boundary

The control plane does not perform network I/O or retries. It exposes
`GatewayAttemptState` for a future transport integration. Retry is denied by
default. Automatic retry is permitted only when:

- the transport explicitly sets `replay_authorized=True` after proving replay is
  safe for this request;
- the transport explicitly records that no upstream side effect is possible;
- at least one attempt has completed;
- another distinct same-provider target remains;
- no response bytes have started;
- no tool call has started.

This prevents a transport from treating an unknown outcome, partial stream, tool
invocation, possibly accepted request, duplicate target, or exhausted attempt
list as safely replayable.

## Accounting boundary

`observe_response` accepts usage only when both conditions hold:

1. the observed provider equals the plan's source provider;
2. the observed provider/model pair exists in the executable attempt list.

Usage from an unplanned target is rejected instead of being silently attributed
to the run.

## Evidence

The executable evidence is in `tests/test_gateway_control_plane.py`.

| Invariant | Test evidence |
| --- | --- |
| Adapter-detected source must match the current candidate | `test_source_provider_must_match_current_candidate` |
| Duplicate target keys cannot inflate retry capacity | `test_duplicate_provider_targets_are_rejected` |
| Cross-provider targets never become executable | `test_cheaper_cross_provider_target_is_excluded_by_policy` |
| Explicit model forcing cannot cross the provider boundary | `test_forced_model_cannot_bypass_cross_provider_boundary` |
| Provider failure does not escape to another provider | `test_provider_failure_does_not_escape_to_another_provider` |
| Same-provider optimization remains available | `test_same_provider_routed_target_is_first_executable_attempt` |
| Retry is denied without explicit replay proof | `test_automatic_retry_is_denied_without_explicit_replay_authorization` |
| Usage from a different provider is rejected | `test_observation_rejects_a_different_provider` |
| Redaction, routing, cache observation, and ledger composition remain intact | `test_control_plane_composes_policy_routing_cache_and_ledger` |

Run the focused evidence suite with:

```bash
python -m pytest -q tests/test_gateway_control_plane.py tests/test_provider_policy.py
python -m ruff check entroly/gateway_control_plane.py tests/test_gateway_control_plane.py
```

Before merging a live proxy integration, also run the proxy transport, provider,
streaming, tool-call, and accounting suites. The focused tests above prove the
control-plane policy only; they do not prove the complete HTTP execution path.

## Claims this evidence supports

The repository may claim that:

- gateway planning requires an explicit source-provider identity;
- executable gateway plans are same-provider by construction;
- automatic cross-provider failover is not implemented by this control plane;
- cost scoring and forced-model input cannot override the provider boundary;
- provider failure fails closed rather than changing the data recipient;
- duplicate targets cannot create artificial retry capacity;
- retry is denied by default and requires explicit replay-safety proof;
- response accounting rejects provider/model pairs outside the execution plan.

## Claims this evidence does not support

This evidence does **not** establish:

- live connectivity to every provider;
- universal compatibility across provider APIs;
- legal compliance for every operator, jurisdiction, contract, or data class;
- permission to use consumer subscription credentials as API credentials;
- safe cross-provider transformation of tools, schemas, vision, reasoning, cache
  controls, or streaming events;
- production readiness of a future `PromptCompilerProxy` integration.

Those claims require separate provider-specific conformance, credential,
contract, data-residency, and end-to-end transport evidence.

## Integration requirements

A future `PromptCompilerProxy` integration must preserve these rules:

1. Run the gateway in shadow mode before it becomes authoritative.
2. Keep one routing authority; do not apply both legacy routing and gateway
   routing to the same request.
3. Pass the adapter-detected provider explicitly as `source_provider`.
4. Keep that source provider fixed for the entire execution plan.
5. Preserve the existing provider-specific request body and headers.
6. Never forward credentials through redirects or to a different origin.
7. Deny retry unless replay safety is explicitly proven in
   `GatewayAttemptState` and another distinct target remains.
8. Assign a distinct attempt identifier to every billable attempt.
9. Record the selected model, route reason, status, latency, usage, and cost.
10. Fail closed when capability, target, usage, or policy evidence is missing.
11. Do not describe the feature as cross-provider failover or universal provider
    compliance.
