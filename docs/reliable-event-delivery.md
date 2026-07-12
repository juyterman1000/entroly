# Reliable Event Delivery

Entroly's Slack, Discord, and Telegram adapters are operational notification
channels, not full conversational messaging platforms. Their job is to deliver
Entroly events reliably and make failure visible.

## Guarantees

Outbound events are written to a local SQLite queue before network I/O. The
queue provides:

- stable idempotency keys for daemon-derived events;
- restart replay for pending deliveries;
- bounded exponential backoff with deterministic jitter;
- provider `Retry-After` support where available;
- short delivery leases so concurrent gateway processes do not normally claim
  the same event;
- dead-letter state after a bounded number of attempts;
- explicit dead-letter retry;
- cryptographic delivery receipts for successful sends;
- bounded, redacted payload persistence;
- no persisted webhook URLs, bot tokens, chat IDs, credentials, or provider
  response bodies.

The default database is:

```text
.entroly/event-delivery.sqlite3
```

Set `ENTROLY_DELIVERY_DB` to place it elsewhere.

## Public API

Each gateway keeps its existing `send()` method and adds an optional stable key:

```python
result = gateway.send(
    "deployment completed",
    event_key="deployment:prod:2026-07-11T23:00Z",
)
```

Reusing the same key for the same channel and destination returns the existing
event. Reusing it with a different payload raises an `IdempotencyConflict`
instead of silently mutating history.

Gateways also expose:

```python
gateway.flush()
gateway.delivery_stats()
```

The lower-level store supports:

```python
store.retry_dead(event_id)
store.prune_delivered(older_than_s=30 * 86400)
```

## Delivery receipts

A successful delivery stores an `entroly.event-delivery-receipt.v1` receipt
containing only:

- event ID;
- channel;
- hashed destination identity;
- payload digest;
- total attempt count;
- delivery time;
- bounded provider status;
- receipt SHA-256 digest.

`DeliveryReceipt.verify()` recomputes the digest locally. The receipt does not
contain message text or credentials.

## Failure semantics

A failed send remains `pending` until its next due time. Once attempts reach the
configured maximum, the event becomes `dead`; it is not silently deleted.
Operators can inspect aggregate counts through `delivery_stats()` and explicitly
requeue a dead event by ID.

Delivery is designed to be effectively-once for Entroly's bounded network calls
and stable event keys. Like most webhook systems, a provider can accept a message
and lose the response before Entroly records success; that narrow ambiguity can
still produce a duplicate after retry. Provider-native idempotency keys should be
used when a channel exposes them.

## Scope boundary

This foundation deliberately does not implement user chat, WhatsApp/Signal
routing, agent steering, durable inbound adoption, or a general omnichannel
message bus. Entroly remains a context, verification, memory, and operational
control layer.
