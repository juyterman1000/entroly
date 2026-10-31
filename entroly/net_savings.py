"""Conservative net saving for one provider-bound request.

Every dollar Entroly shows as *net saving* must survive the question "how could
this number lie?". This module answers it by construction:

* **Net, not gross.** Content Entroly adds (injected context, rewritten blocks)
  is charged, and so is the provider cache prefix its edits forced the
  provider to write again.
* **The provider's own cache discount is never credited.** Only differences
  between the request the client built and the request Entroly forwarded are
  priced.
* **Every unknown is resolved against Entroly.** Removed content is priced at
  the cheapest rate it could ever have been billed at, added content at the
  most expensive, and the token conversion that would favour Entroly is never
  the one used.
* **Zero when nothing changed.** If the forwarded prompt is the prompt the
  client built, the estimate is exactly zero.
* **Negative is a result.** A request where Entroly cost more is reported as
  a negative saving and is never clamped.

Pricing for one request with provider usage ``U`` (the authority for how many
tokens were billed) and catalog rates for the executed model:

* ``r_read``: cache-read rate, the cheapest an input token can be billed.
* ``r_in``: uncached input rate.
* ``r_write``: the cache-write rate of the write a charged token came from:
  1-hour or 5-minute when the provider reports the split, otherwise the
  1-hour rate, the dearest a write can be.

Units are prompt blocks (system text, each tool, each message or content
block) identified only by a truncated SHA-256 digest and a byte length. The
multiset difference between the client's blocks and the forwarded blocks
yields ``removed`` and ``added`` bytes; a rewritten block counts as both.

Byte counts become provider tokens through the provider's own count of the
forwarded request, ``T = U.total_input_tokens``:

* removed bytes use ``T / surface_bytes`` (the canonical surface includes JSON
  framing, so this ratio is the smaller one and under-counts removed tokens);
* added and re-written bytes use ``T / unit_bytes`` (the larger ratio, so it
  over-counts what Entroly added).

Net saving::

    S = removed_seen  * r_read            # may already have been cached
      + removed_new   * r_in              # never sent before: not cacheable
      - added         * r_write           # what Entroly put in the request
      - interference  * (r_write - r_read)  # cached prefix Entroly broke

The two charges are bounded by provider truth: only the forwarded tokens the
provider did *not* serve from cache (uncached + cache-write) can have cost more
than a cache read. That pool is assigned to the added tokens first, then to the
broken prefix, dearest tokens first (1-hour writes, 5-minute writes, uncached
input at ``r_in``), and any added tokens beyond it are charged at ``r_read``. On a
later turn, content Entroly already forwarded is mostly a cache read, and is
charged as one, instead of turning every follow-up turn into a false loss.

``removed_new`` covers blocks that no recent request through this proxy has
sent (any conversation, since subagent forks share history). The provider
cache can only serve a prefix that was written before, so a block never sent
cannot be a cache read. A block seen before is priced as a cache read.

Stated limits, carried with every estimate:

* input side only; output tokens are not counted;
* bytes per token of removed content is assumed comparable to the forwarded
  request (bounded by the two ratios above, not eliminated);
* on a request that rewrites old history, removed bytes can also appear in
  the interference charge, so that request's estimate is lower than reality
  (later requests in the conversation recover the saving).

Only digests, byte counts and integer microdollars are produced. No prompt
content is stored and nothing here makes a network call.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

BASIS = "provider-calibrated estimate"
_BUNDLED_PRICING = Path(__file__).resolve().parent / "data" / "anthropic_pricing.json"
_DIGEST_HEX = 32  # 128-bit truncated SHA-256: identity only, never content
_SEEN_CAPACITY = 200_000

_PROMPT_FIELDS = ("system", "instructions", "systemInstruction", "tools", "messages", "input", "contents")


# ── pricing ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TierRates:
    """USD per one million tokens for one billing tier."""

    input: Decimal
    cache_read: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal

    @property
    def cache_write_max(self) -> Decimal:
        return max(self.cache_write_5m, self.cache_write_1h)


@dataclass(frozen=True, slots=True)
class ModelRates:
    base: TierRates
    long_context: TierRates | None = None
    long_context_threshold: int | None = None

    def tier(self, total_input_tokens: int) -> TierRates:
        if (
            self.long_context is not None
            and self.long_context_threshold is not None
            and total_input_tokens > self.long_context_threshold
        ):
            return self.long_context
        return self.base


@dataclass(frozen=True, slots=True)
class NetSavingCatalog:
    entries: Mapping[str, ModelRates]
    source: str

    def resolve(self, provider: str, model: str) -> ModelRates | None:
        return self.entries.get(f"{provider.lower().strip()}:{model.strip()}")


def _rate(raw: Mapping[str, Any], key: str) -> Decimal:
    value = Decimal(str(raw[key]))
    if value < 0:
        raise ValueError(f"{key} cannot be negative")
    return value


def _tier(raw: Mapping[str, Any]) -> TierRates:
    five = _rate(raw, "cache_write_per_million")
    one = (
        _rate(raw, "cache_write_1h_per_million")
        if raw.get("cache_write_1h_per_million") is not None
        else five
    )
    return TierRates(
        input=_rate(raw, "input_per_million"),
        cache_read=_rate(raw, "cache_read_per_million"),
        cache_write_5m=five,
        cache_write_1h=one,
    )


def catalog_from_mapping(payload: Mapping[str, Any], *, source: str) -> NetSavingCatalog:
    models = payload.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("pricing catalog requires a non-empty models mapping")
    entries: dict[str, ModelRates] = {}
    for key, raw in models.items():
        provider, sep, model = str(key).partition(":")
        if not sep or not provider or not model or not isinstance(raw, Mapping):
            raise ValueError(f"invalid pricing entry {key!r}")
        if "cache_write_per_million" not in raw:
            # An explicit operator catalog may omit cache writes; the ledger
            # then bills writes at the input rate. Mirror that, never lower.
            raw = {**raw, "cache_write_per_million": raw["input_per_million"]}
        long_raw = raw.get("long_context")
        long_tier = None
        threshold = None
        if isinstance(long_raw, Mapping):
            long_tier = _tier(long_raw)
            threshold = int(long_raw["threshold_input_tokens"])
        entries[f"{provider.lower()}:{model}"] = ModelRates(
            base=_tier(raw), long_context=long_tier, long_context_threshold=threshold
        )
    return NetSavingCatalog(entries=entries, source=source)


_CATALOG_LOCK = threading.Lock()
_CATALOG_CACHE: dict[str, NetSavingCatalog | None] = {}


def bundled_catalog() -> NetSavingCatalog | None:
    """The bundled, source-linked Anthropic table, or None when disabled."""
    if os.environ.get("ENTROLY_BUNDLED_PRICING", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    with _CATALOG_LOCK:
        if "bundled" not in _CATALOG_CACHE:
            try:
                payload = json.loads(_BUNDLED_PRICING.read_text(encoding="utf-8"))
                _CATALOG_CACHE["bundled"] = catalog_from_mapping(
                    payload, source=str(payload.get("source") or "bundled")
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                _CATALOG_CACHE["bundled"] = None
        return _CATALOG_CACHE["bundled"]


def explicit_catalog(path: str) -> NetSavingCatalog | None:
    """The operator's ``ENTROLY_PRICING_CATALOG`` file, read for rates only."""
    with _CATALOG_LOCK:
        key = f"file:{path}"
        if key not in _CATALOG_CACHE:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                _CATALOG_CACHE[key] = catalog_from_mapping(
                    payload, source=str(payload.get("source") or key)
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                _CATALOG_CACHE[key] = None
        return _CATALOG_CACHE[key]


def resolve_rates(provider: str, model: str) -> tuple[ModelRates, str] | None:
    """Operator catalog first; the bundled table only when none is configured."""
    path = os.environ.get("ENTROLY_PRICING_CATALOG", "").strip()
    catalog = explicit_catalog(path) if path else bundled_catalog()
    if catalog is None:
        return None
    rates = catalog.resolve(provider, model)
    return (rates, catalog.source) if rates is not None else None


def _reset_catalog_cache_for_tests() -> None:
    with _CATALOG_LOCK:
        _CATALOG_CACHE.clear()


# ── content-blind prompt inventory ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PromptInventory:
    """Digest and byte length of every prompt block; no content."""

    units: tuple[tuple[str, int], ...]
    unit_bytes: int
    surface_bytes: int


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8", "surrogatepass")


def _unit(field: str, value: Any) -> tuple[str, int]:
    encoded = _canonical([field, value])
    return hashlib.sha256(encoded).hexdigest()[:_DIGEST_HEX], len(encoded)


def _units_for(field: str, value: Any) -> Iterable[tuple[str, int]]:
    if field in {"tools", "system", "instructions"} and isinstance(value, list):
        for item in value:
            yield _unit(field, item)
        return
    if field in {"messages", "input", "contents"} and isinstance(value, list):
        for message in value:
            if not isinstance(message, Mapping):
                yield _unit(field, message)
                continue
            content = message.get("content", message.get("parts"))
            if isinstance(content, list):
                role = message.get("role", "")
                for block in content:
                    yield _unit(f"{field}:{role}", block)
            else:
                yield _unit(field, message)
        return
    yield _unit(field, value)


def prompt_inventory(body: Mapping[str, Any]) -> PromptInventory:
    units: list[tuple[str, int]] = []
    present = [(field, body[field]) for field in _PROMPT_FIELDS if field in body]
    for field, value in present:
        units.extend(_units_for(field, value))
    return PromptInventory(
        units=tuple(units),
        unit_bytes=sum(length for _, length in units),
        surface_bytes=len(_canonical(present)),
    )


def inventory_delta(
    raw: PromptInventory, outbound: PromptInventory
) -> tuple[tuple[tuple[str, int], ...], int]:
    """Removed units (digest, bytes) and added bytes, as a multiset difference."""
    raw_count = Counter(raw.units)
    out_count = Counter(outbound.units)
    removed: list[tuple[str, int]] = []
    for unit, count in (raw_count - out_count).items():
        removed.extend([unit] * count)
    added = sum(length * count for (_, length), count in (out_count - raw_count).items())
    return tuple(sorted(removed)), added


class SeenUnits:
    """Bounded, process-wide set of block digests recently sent upstream.

    Process-wide rather than per conversation: Claude Code subagents and side
    requests re-send another conversation's history, and the provider cache is
    shared across them. A block counts as possibly cached once any request has
    carried it.
    """

    def __init__(self, capacity: int = _SEEN_CAPACITY) -> None:
        self._capacity = max(1, int(capacity))
        self._digests: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def classify_and_remember(
        self,
        removed: Iterable[tuple[str, int]],
        raw: PromptInventory,
        outbound: PromptInventory,
    ) -> tuple[int, int]:
        """Return (seen_bytes, new_bytes) of removed units, then record the request."""
        seen = new = 0
        with self._lock:
            for digest, length in removed:
                if digest in self._digests:
                    seen += length
                else:
                    new += length
            for digest, _ in (*raw.units, *outbound.units):
                self._digests[digest] = None
                self._digests.move_to_end(digest)
            while len(self._digests) > self._capacity:
                self._digests.popitem(last=False)
        return seen, new

    def clear(self) -> None:
        with self._lock:
            self._digests.clear()


SEEN = SeenUnits()


# ── the estimate ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NetSavingEstimate:
    micro_usd: int
    removed_seen_tokens: int
    removed_new_tokens: int
    added_tokens: int
    interference_tokens: int
    pricing_source: str
    long_context_tier: bool

    def details(self) -> dict[str, int | bool]:
        return {
            "removed_seen_tokens": self.removed_seen_tokens,
            "removed_new_tokens": self.removed_new_tokens,
            "added_tokens": self.added_tokens,
            "interference_tokens": self.interference_tokens,
            "long_context_tier": self.long_context_tier,
        }


def _tokens(nbytes: int, total_tokens: int, basis_bytes: int, *, round_up: bool) -> int:
    if nbytes <= 0 or total_tokens <= 0 or basis_bytes <= 0:
        return 0
    numerator = nbytes * total_tokens
    if round_up:
        return -(-numerator // basis_bytes)
    return numerator // basis_bytes


def _micro(tokens: int, rate_per_million: Decimal) -> Decimal:
    # Tokens x USD/MTok = microdollars exactly; rounding happens once, at the end.
    return Decimal(tokens) * rate_per_million


def _draw(
    pool: list[list[Any]], tokens: int, read_rate: Decimal, *, minus_read: bool
) -> tuple[Decimal, int]:
    """Charge ``tokens`` against the dearest remaining pool tokens.

    Returns the microdollar charge and how many tokens the pool could not
    cover (those were cache reads). ``minus_read`` charges only the premium
    over a cache read -- the cost of a prefix that should have been read.
    """
    cost = Decimal(0)
    remaining = max(0, int(tokens))
    for slot in pool:
        if remaining <= 0:
            break
        take = min(remaining, int(slot[0]))
        if take <= 0:
            continue
        rate = slot[1] - read_rate if minus_read else slot[1]
        cost += _micro(take, max(Decimal(0), rate))
        slot[0] = int(slot[0]) - take
        remaining -= take
    return cost, remaining


def estimate_net_saving(
    *,
    raw: PromptInventory,
    outbound: PromptInventory,
    removed_seen_bytes: int,
    removed_new_bytes: int,
    added_bytes: int,
    interference_bytes: int,
    uncached_input_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    rates: ModelRates,
    pricing_source: str,
    cache_write_1h_tokens: int | None = None,
) -> NetSavingEstimate:
    """Price the difference between the client's prompt and the forwarded one."""
    if raw.units == outbound.units and interference_bytes <= 0:
        return NetSavingEstimate(0, 0, 0, 0, 0, pricing_source, False)

    uncached = max(0, int(uncached_input_tokens))
    reads = max(0, int(cache_read_tokens))
    writes = max(0, int(cache_write_tokens))
    total = uncached + reads + writes
    tier = rates.tier(total)
    # Forwarded tokens the provider did NOT serve from cache are the only ones
    # that can have cost more than a cache read. They form a priced pool, and
    # Entroly's charges are drawn from its dearest tokens first: 1-hour writes,
    # then 5-minute writes, then uncached input. Without the provider's TTL
    # split, every write is assumed to be a 1-hour write.
    one_hour = writes if cache_write_1h_tokens is None else min(writes, max(0, int(cache_write_1h_tokens)))
    pool = sorted(
        [
            [one_hour, tier.cache_write_1h],
            [writes - one_hour, tier.cache_write_5m],
            [uncached, tier.input],
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    # Smaller tokens-per-byte ratio for what Entroly removed, larger for what
    # it added or broke: any conversion error lands against Entroly.
    seen_tokens = _tokens(removed_seen_bytes, total, outbound.surface_bytes, round_up=False)
    new_tokens = _tokens(removed_new_bytes, total, outbound.surface_bytes, round_up=False)
    added_tokens = _tokens(added_bytes, total, outbound.unit_bytes, round_up=True)
    broken_tokens = _tokens(interference_bytes, total, outbound.unit_bytes, round_up=True)

    # Removed content priced at the base tier: the counterfactual's own (often
    # higher) long-context tier is not assumed. The actual request's tier sets
    # the cost of what Entroly added.
    added_cost, added_left = _draw(pool, added_tokens, tier.cache_read, minus_read=False)
    broken_cost, _ = _draw(pool, broken_tokens, tier.cache_read, minus_read=True)
    saving = (
        _micro(seen_tokens, rates.base.cache_read)
        + _micro(new_tokens, rates.base.input)
        - added_cost
        - _micro(added_left, tier.cache_read)
        - broken_cost
    )
    return NetSavingEstimate(
        micro_usd=int(saving.to_integral_value(rounding=ROUND_FLOOR)),
        removed_seen_tokens=seen_tokens,
        removed_new_tokens=new_tokens,
        added_tokens=added_tokens,
        interference_tokens=broken_tokens,
        pricing_source=pricing_source,
        long_context_tier=tier is not rates.base,
    )


# ── the governor ────────────────────────────────────────────────────────

GOVERNOR_HEADER = "X-Entroly-Net-Saving-Governor"
_GOVERNOR_CAPACITY = 4_096
_DEFAULT_FLOOR_MICRO_USD = 20_000  # $0.02
_DEFAULT_MIN_REQUESTS = 3


def _env_flag_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, "").strip()))
    except ValueError:
        return default


def _env_floor_micro_usd() -> int:
    raw = os.environ.get("ENTROLY_NET_SAVING_FLOOR_USD", "").strip()
    if not raw:
        return _DEFAULT_FLOOR_MICRO_USD
    try:
        value = Decimal(raw)
    except ArithmeticError:
        return _DEFAULT_FLOOR_MICRO_USD
    if not value.is_finite() or value < 0:
        return _DEFAULT_FLOOR_MICRO_USD
    return int((value * 1_000_000).to_integral_value(rounding=ROUND_FLOOR))


@dataclass(frozen=True)
class GovernorPolicy:
    enabled: bool = True
    floor_micro_usd: int = _DEFAULT_FLOOR_MICRO_USD
    min_requests: int = _DEFAULT_MIN_REQUESTS

    @classmethod
    def from_env(cls) -> GovernorPolicy:
        return cls(
            enabled=not _env_flag_off("ENTROLY_NET_SAVING_GOVERNOR"),
            floor_micro_usd=_env_floor_micro_usd(),
            min_requests=_env_int(
                "ENTROLY_NET_SAVING_MIN_REQUESTS", _DEFAULT_MIN_REQUESTS, minimum=1
            ),
        )


@dataclass(frozen=True)
class GovernorStanding:
    """What the governor knows about one conversation, for receipts."""

    action: str  # "hold" or "observe"
    priced_requests: int
    cumulative_micro_usd: int
    floor_micro_usd: int
    min_requests: int

    def details(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "priced_requests": self.priced_requests,
            "cumulative_micro_usd": self.cumulative_micro_usd,
            "floor_micro_usd": self.floor_micro_usd,
            "min_requests": self.min_requests,
        }


class NetSavingGovernor:
    """Stops Entroly's optional optimizations where they are not paying.

    Per conversation it adds up the same signed, conservative net saving that
    receipts report. Once a conversation has at least ``min_requests`` priced
    requests and its cumulative saving is below ``-floor``, it *holds*: the
    proxy forwards the client's own request instead of Entroly's optional
    rewrites for the rest of that conversation. The hold latches, so the
    decision depends only on receipts already sealed, never on one that is
    still in flight.

    Requests without priced evidence add nothing, so conversations Entroly
    cannot price are never held. Emergency session rescue and required
    redaction are outside its reach: rescue is part of the baseline it
    returns to, and redaction is applied after the choice.

    State is a bounded LRU keyed by a digest of the conversation id, holding
    two integers and a flag. No prompt content is kept.
    """

    def __init__(self, capacity: int = _GOVERNOR_CAPACITY) -> None:
        self._capacity = max(1, int(capacity))
        self._entries: OrderedDict[str, list[int]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(conversation_id: str) -> str:
        return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:_DIGEST_HEX]

    def _standing(self, entry: list[int], policy: GovernorPolicy) -> GovernorStanding:
        count, total, held = entry
        return GovernorStanding(
            action="hold" if held else "observe",
            priced_requests=count,
            cumulative_micro_usd=total,
            floor_micro_usd=policy.floor_micro_usd,
            min_requests=policy.min_requests,
        )

    def record(
        self,
        conversation_id: str,
        micro_usd: int | None,
        *,
        policy: GovernorPolicy | None = None,
    ) -> GovernorStanding | None:
        """Add one sealed receipt's estimate; ``None`` estimates add nothing."""
        policy = policy or GovernorPolicy.from_env()
        if not policy.enabled or not conversation_id or micro_usd is None:
            return None
        key = self._key(conversation_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = [0, 0, 0]
                self._entries[key] = entry
            self._entries.move_to_end(key)
            entry[0] += 1
            entry[1] += int(micro_usd)
            if entry[0] >= policy.min_requests and entry[1] < -policy.floor_micro_usd:
                entry[2] = 1
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
            return self._standing(entry, policy)

    def standing(
        self, conversation_id: str, *, policy: GovernorPolicy | None = None
    ) -> GovernorStanding | None:
        """The current decision for a conversation; ``None`` when disabled."""
        policy = policy or GovernorPolicy.from_env()
        if not policy.enabled or not conversation_id:
            return None
        with self._lock:
            entry = self._entries.get(self._key(conversation_id))
            if entry is None:
                return GovernorStanding("observe", 0, 0, policy.floor_micro_usd,
                                        policy.min_requests)
            self._entries.move_to_end(self._key(conversation_id))
            return self._standing(entry, policy)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


GOVERNOR = NetSavingGovernor()


__all__ = [
    "BASIS",
    "GOVERNOR",
    "GOVERNOR_HEADER",
    "GovernorPolicy",
    "GovernorStanding",
    "ModelRates",
    "NetSavingCatalog",
    "NetSavingEstimate",
    "NetSavingGovernor",
    "PromptInventory",
    "SEEN",
    "SeenUnits",
    "TierRates",
    "bundled_catalog",
    "catalog_from_mapping",
    "estimate_net_saving",
    "explicit_catalog",
    "inventory_delta",
    "prompt_inventory",
    "resolve_rates",
]
