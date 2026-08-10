"""Coding tasks that genuinely require the context to be present.

Two designs were rejected before this one, both by measurement rather than
argument:

1. **Guessable bugs.** `assert is_adult(18) is True` states the fix, so a null
   arm with no context at all solved 4 of 4 -- as did the raw and compressed
   arms. Three arms at 100% measures nothing.

2. **Arbitrary constants pinned by the test.** Moving the answer into another
   file does not help if the test reveals it: `assert quote('eu', 100) == 121`
   yields the multiplier by division, and any boundary assertion
   (`should_retry(7) is False`) states its own boundary.

What survives is a dependency whose **shape** contradicts the obvious
assumption. The import line is already visible in the buggy file, so a model
can always reference a symbol; what it cannot invent is that the lookup is
keyed by an uppercase code, or that the helper takes a different unit than its
name suggests. Those are ordinary integration bugs, and reading the dependency
is the only way to resolve them.

Each task therefore has:

  * a bug in one file, whose fix must call into a second file;
  * a second file whose interface is not what a reasonable person would guess;
  * a test that asserts behaviour without disclosing the shape.

This makes the set a real probe of dependency closure -- keeping the buggy file
while dropping its dependency is exactly the failure mode worth detecting, and
one this engine has been observed to produce (`optimize()` returning a caller
and dropping its callee while using 3% of the budget).

The null arm is permanent. If it ever passes, these tasks have decayed into
guessable ones and every comparison built on them is void.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DependentTask:
    """A bug whose fix requires reading a second file."""

    task_id: str
    query: str
    target_file: str
    broken_source: str
    dependency_file: str
    dependency_source: str
    test_file: str
    test_source: str
    #: What a solver must have learned from the dependency. Never shown to the
    #: model; used to document why the task is not guessable.
    hidden_shape: str
    distractors: dict[str, str] = field(default_factory=dict)

    def context_files(self) -> dict[str, str]:
        files = {
            self.target_file: self.broken_source,
            self.dependency_file: self.dependency_source,
        }
        files.update(self.distractors)
        return files


def _distractors() -> dict[str, str]:
    """Plausible, irrelevant modules -- the bulk of any real repository."""
    return {
        "notifications/mailer.py": (
            "TEMPLATES = {'welcome': 'Hello {name}', 'reset': 'Reset: {link}'}\n\n"
            "def render(name, **fields):\n"
            "    return TEMPLATES[name].format(**fields)\n"
        ),
        "inventory/warehouse.py": (
            "class Warehouse:\n"
            "    def __init__(self):\n"
            "        self._stock = {}\n\n"
            "    def receive(self, sku, quantity):\n"
            "        self._stock[sku] = self._stock.get(sku, 0) + quantity\n"
            "        return self._stock[sku]\n"
        ),
        "geo/routing.py": (
            "import math\n\n"
            "def haversine(a, b):\n"
            "    radius = 6371.0\n"
            "    dlat = math.radians(b[0] - a[0])\n"
            "    return 2 * radius * math.asin(math.sqrt(math.sin(dlat / 2) ** 2))\n"
        ),
        "reporting/exporter.py": (
            "import csv\n\n"
            "def export_rows(path, rows, headers):\n"
            "    with open(path, 'w', newline='') as handle:\n"
            "        writer = csv.DictWriter(handle, fieldnames=headers)\n"
            "        writer.writeheader()\n"
            "        writer.writerows(rows)\n"
            "    return len(rows)\n"
        ),
        "analytics/metrics.py": (
            "def percentile(values, fraction):\n"
            "    if not values:\n"
            "        return 0.0\n"
            "    ordered = sorted(values)\n"
            "    return ordered[int(round(fraction * (len(ordered) - 1)))]\n"
        ),
        "billing/ledger.py": (
            "def post_entry(book, account, amount):\n"
            "    book.setdefault(account, []).append(amount)\n"
            "    return sum(book[account])\n"
        ),
    }


def build_dependent_tasks(distractor_count: int = 6) -> list[DependentTask]:
    shared = dict(list(_distractors().items())[:distractor_count])

    return [
        DependentTask(
            task_id="uppercase_region_key",
            query="quote() raises KeyError for a lowercase region code",
            target_file="core/pricing.py",
            broken_source=(
                "from config.rates import REGION_MULTIPLIER\n\n\n"
                "def quote(region, base):\n"
                "    \"\"\"Total for a region, rounded to the nearest whole unit.\"\"\"\n"
                "    return round(base * REGION_MULTIPLIER[region])\n"
            ),
            dependency_file="config/rates.py",
            dependency_source=(
                "# Rates are keyed by ISO region code, always upper case.\n"
                "REGION_MULTIPLIER = {\n"
                "    'US': 1.07,\n"
                "    'EU': 1.21,\n"
                "    'APAC': 1.13,\n"
                "}\n"
            ),
            test_file="test_pricing.py",
            test_source=(
                "from config.rates import REGION_MULTIPLIER\n"
                "from core.pricing import quote\n\n"
                "def test_quote_accepts_lowercase():\n"
                "    expected = round(100 * REGION_MULTIPLIER['EU'])\n"
                "    assert quote('eu', 100) == expected\n"
                "    assert quote('EU', 100) == expected\n"
            ),
            hidden_shape="REGION_MULTIPLIER is keyed by UPPER-case codes",
            distractors=shared,
        ),
        DependentTask(
            task_id="helper_takes_minutes",
            query="session_expiry passes the wrong unit to the helper",
            target_file="core/sessions.py",
            broken_source=(
                "from config.timing import to_deadline\n\n\n"
                "def session_expiry(now, ttl_seconds):\n"
                "    \"\"\"Deadline for a session with the given TTL in seconds.\"\"\"\n"
                "    return to_deadline(now, ttl_seconds)\n"
            ),
            dependency_file="config/timing.py",
            dependency_source=(
                "def to_deadline(now, ttl_minutes):\n"
                "    \"\"\"Deadline from a TTL expressed in MINUTES.\"\"\"\n"
                "    return now + ttl_minutes * 60\n"
            ),
            test_file="test_sessions.py",
            test_source=(
                "from core.sessions import session_expiry\n\n"
                "def test_session_expiry_uses_seconds():\n"
                "    assert session_expiry(0, 120) == 120\n"
                "    assert session_expiry(1000, 60) == 1060\n"
            ),
            hidden_shape="to_deadline() takes MINUTES, not seconds",
            distractors=shared,
        ),
        DependentTask(
            task_id="status_enum_values",
            query="is_open does not recognise the project's open statuses",
            target_file="core/tickets.py",
            broken_source=(
                "from config.statuses import OPEN_STATUSES\n\n\n"
                "def is_open(status):\n"
                "    \"\"\"True when the ticket is in an open state.\"\"\"\n"
                "    return status == 'open'\n"
            ),
            dependency_file="config/statuses.py",
            dependency_source=(
                "# Workflow states. 'open' is not one of them; the tracker\n"
                "# splits open work into triage and active.\n"
                "OPEN_STATUSES = frozenset({'triage', 'active', 'blocked'})\n"
                "CLOSED_STATUSES = frozenset({'shipped', 'wontfix'})\n"
            ),
            test_file="test_tickets.py",
            test_source=(
                "from config.statuses import CLOSED_STATUSES, OPEN_STATUSES\n"
                "from core.tickets import is_open\n\n"
                "def test_is_open_matches_workflow():\n"
                "    for status in OPEN_STATUSES:\n"
                "        assert is_open(status) is True\n"
                "    for status in CLOSED_STATUSES:\n"
                "        assert is_open(status) is False\n"
            ),
            hidden_shape="open states are triage/active/blocked, never 'open'",
            distractors=shared,
        ),
        DependentTask(
            task_id="fee_returns_cents",
            query="checkout_total mixes units when adding the fee",
            target_file="core/checkout.py",
            broken_source=(
                "from config.fees import processing_fee\n\n\n"
                "def checkout_total(amount_pounds):\n"
                "    \"\"\"Order total in POUNDS, including the processing fee.\"\"\"\n"
                "    return amount_pounds + processing_fee(amount_pounds)\n"
            ),
            dependency_file="config/fees.py",
            dependency_source=(
                "def processing_fee(amount_pounds):\n"
                "    \"\"\"Fee for an order, returned in PENCE.\"\"\"\n"
                "    return int(round(amount_pounds * 100 * 0.02))\n"
            ),
            test_file="test_checkout.py",
            test_source=(
                "from core.checkout import checkout_total\n\n"
                "def test_checkout_total_is_in_pounds():\n"
                "    assert checkout_total(100) == 102\n"
                "    assert checkout_total(50) == 51\n"
            ),
            hidden_shape="processing_fee() returns PENCE while the caller uses pounds",
            distractors=shared,
        ),
    ]
