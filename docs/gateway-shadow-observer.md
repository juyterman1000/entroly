# Gateway shadow observer

`GatewayShadowObserver` is the non-authoritative integration seam between a live
provider proxy and `GatewayControlPlane`.

## Purpose

The observer answers one question without changing production behavior:

> Can this exact provider request be canonicalized and planned under the gateway
> safety contract?

It returns a bounded `GatewayShadowReceipt`. It does not return a provider body,
credentials, prompt text, tool schemas, or an executable network action.

## Safety properties

- The caller-owned request mapping is never mutated.
- Only the inbound provider and current model are executable candidates.
- Cross-provider routing cannot be introduced through the shadow API.
- Planner failures become `shadow_error` receipts instead of request failures.
- Receipts contain identifiers, capability names, target keys, route reasons,
  timing, and policy outcomes only.
- Headers are bounded and contain no request content.
- The observer performs no network I/O, retry, billing, or usage recording.

## Proxy integration contract

A live proxy may run the observer after provider detection and JSON parsing, but
before request mutation. Integration must follow these rules:

1. Copy or hash the inbound body before observation.
2. Call `observe(...)` with the detected provider, original body, bounded
   headers, path, and request ID.
3. Store the receipt in local metrics or attach the bounded shadow headers.
4. Continue using the existing proxy body, provider, model, routing, retry, and
   accounting paths unchanged.
5. Never feed `planned_model` back into live execution while shadow mode is
   active.
6. Never fail a user request because shadow observation failed.
7. Track disagreement between the existing live route and shadow receipt before
   considering any authoritative integration.

## Evidence

`tests/test_gateway_shadow.py` proves that:

- OpenAI streaming requests remain byte-equivalent at the Python object level;
- receipts and headers do not expose prompt secrets;
- unknown providers fail closed without request mutation;
- Anthropic tool capability evidence is preserved;
- the only executable target is the current provider/model.

Run:

```bash
python -m pytest -q tests/test_gateway_shadow.py tests/test_gateway_control_plane.py
python -m ruff check entroly/gateway_shadow.py tests/test_gateway_shadow.py
```

## Deliberate non-goals

This module does not:

- wire itself into `PromptCompilerProxy`;
- change models or providers;
- execute retries or failover;
- replace existing cache routing;
- record provider usage or cost;
- prove live provider conformance;
- authorize cross-provider routing.

The final live hook should be a small follow-up change after this observer's
focused tests and repository CI are green.
