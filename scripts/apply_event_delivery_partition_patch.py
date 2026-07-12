#!/usr/bin/env python3
"""Apply the exact P3 queue partitioning correction once."""

from pathlib import Path

path = Path("entroly/integrations/event_delivery.py")
text = path.read_text(encoding="utf-8")

old_signature = '''    def claim_due(
        self,
        *,
        limit: int = 32,
        lease_seconds: float = 30.0,
    ) -> list[DeliveryEvent]:
'''
new_signature = '''    def claim_due(
        self,
        *,
        channel: str,
        destination_hash: str,
        limit: int = 32,
        lease_seconds: float = 30.0,
    ) -> list[DeliveryEvent]:
'''
old_query = '''                SELECT event_id FROM delivery_events
                WHERE state = 'pending'
                  AND not_before <= ?
                  AND (lease_until IS NULL OR lease_until <= ?)
                ORDER BY created_at, event_id
                LIMIT ?
                """,
                (now, now, limit),
'''
new_query = '''                SELECT event_id FROM delivery_events
                WHERE channel = ?
                  AND destination_hash = ?
                  AND state = 'pending'
                  AND not_before <= ?
                  AND (lease_until IS NULL OR lease_until <= ?)
                ORDER BY created_at, event_id
                LIMIT ?
                """,
                (channel, destination_hash, now, now, limit),
'''
old_dispatch = '''            for event in self.store.claim_due(
                limit=limit,
                lease_seconds=self._lease_seconds,
            ):
'''
new_dispatch = '''            for event in self.store.claim_due(
                channel=self.channel,
                destination_hash=self.destination_hash,
                limit=limit,
                lease_seconds=self._lease_seconds,
            ):
'''

for old, new, label in (
    (old_signature, new_signature, "claim signature"),
    (old_query, new_query, "claim query"),
    (old_dispatch, new_dispatch, "dispatcher claim"),
):
    if new in text:
        continue
    if old not in text:
        raise SystemExit(f"refusing to patch: missing {label}")
    text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
