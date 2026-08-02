# Provider conformance

Entroly exposes an offline, executable provider-conformance check:

```bash
python -m entroly.provider_conformance --json
```

The report distinguishes two different guarantees:

- **Same-provider model rewrites** preserve the original provider body, including unknown and provider-specific controls.
- **Cross-provider rendering** is limited to text-only portable messages. Tools, structured-output schemas, vision, reasoning controls, cache controls, non-text message blocks, and unmapped provider fields fail closed instead of being silently dropped.

The command makes no network requests and does not claim provider connectivity or complete semantic equivalence. Live provider connectivity and streaming behavior require separate integration tests with operator-supplied credentials.
