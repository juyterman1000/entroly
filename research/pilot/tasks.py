"""Frozen continuity task suite.

Each task is a small real repository with a seeded defect or unfinished feature,
an independent pytest verifier, and a deterministic phase-1 state representing
what agent A did before interruption.

Phase 1 is scripted rather than improvised. That costs some realism and buys
three things the pilot needs more: every arm resumes from a byte-identical tree,
the interruption point is the same for all arms, and the recorded facts are fixed
so B1 and B3 describe the *same* truth rather than two different stories.

`verifier` is run by the harness, never by either agent, and neither agent's
self-report is consulted.

`recorded` is the state agent A left behind. It is the single source for both
baselines: B1 renders it as prose, B3 records it through the real Entroly APIs.
Anything not in `recorded` is unknown to every arm -- no arm gets a fact the
others cannot have.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    task_id: str
    stratum: str
    # The original instruction, identical in every arm.
    statement: str
    # path -> content at the checkpoint (already includes agent A's partial work)
    files: dict[str, str]
    # Command the harness runs to decide verified completion. Exit 0 == success.
    verifier: tuple[str, ...]
    recorded: "RecordedState" = field(default=None)  # type: ignore[assignment]


@dataclass(frozen=True)
class RecordedState:
    """What agent A actually left behind, in the graph's own vocabulary."""

    branch: str
    task_title: str
    remaining_work: tuple[str, ...]
    decisions: tuple[str, ...]
    # ("name", "failed"/"passed", "source_ref")
    verifications: tuple[tuple[str, str, str], ...]
    changed_paths: tuple[str, ...]
    # Approaches agent A tried and rejected. Recorded as decisions, because that
    # is where the graph puts explicitly recorded reasoning; a rejected approach
    # is a decision not to take it.
    rejected: tuple[str, ...] = ()


# ── S1: single-file / local ────────────────────────────────────────────

T01 = Task(
    task_id="t01-rounding",
    stratum="simple_local",
    statement=(
        "money.py must round half-up to 2 decimal places for currency amounts. "
        "Make tests/test_money.py pass."
    ),
    files={
        "money.py": (
            "from decimal import Decimal\n\n\n"
            "def to_cents(amount):\n"
            "    # Partial: Decimal conversion done, rounding mode not chosen yet.\n"
            "    return Decimal(str(amount))\n"
        ),
        "tests/test_money.py": (
            "from decimal import Decimal\n\n"
            "from money import to_cents\n\n\n"
            "def test_half_up():\n"
            "    assert to_cents(2.675) == Decimal('2.68')\n"
            "    assert to_cents(2.665) == Decimal('2.67')\n"
            "    assert to_cents(1.005) == Decimal('1.01')\n\n\n"
            "def test_exact_values_unchanged():\n"
            "    assert to_cents(3.10) == Decimal('3.10')\n"
        ),
    },
    verifier=("python", "-m", "pytest", "-q", "tests/test_money.py"),
    recorded=RecordedState(
        branch="feat/money-rounding",
        task_title="round currency amounts half-up to two places",
        remaining_work=(
            "apply ROUND_HALF_UP quantisation in to_cents",
        ),
        decisions=(
            "amounts must be Decimal end to end; float rounding was the original bug",
        ),
        verifications=(("money rounding tests", "failed",
                        "pytest tests/test_money.py"),),
        changed_paths=("money.py",),
        rejected=(
            "python round() was tried and rejected: it is banker's rounding, so "
            "2.675 goes to 2.67 and the first assertion still fails",
        ),
    ),
)

# ── S2: multi-file ─────────────────────────────────────────────────────

T02 = Task(
    task_id="t02-retry-backoff",
    stratum="multi_file",
    statement=(
        "client.py should retry transient failures using the policy in "
        "backoff.py. Make tests/test_client.py pass."
    ),
    files={
        "backoff.py": (
            "def delays(attempts, base=0.1):\n"
            '    """Exponential backoff delays, newest API."""\n'
            "    return [base * (2 ** i) for i in range(attempts)]\n"
        ),
        "client.py": (
            "from backoff import delays\n\n\n"
            "class Transient(Exception):\n"
            "    pass\n\n\n"
            "def call(fn, attempts=3):\n"
            "    # Partial: backoff imported, retry loop not written yet.\n"
            "    return fn()\n"
        ),
        "tests/test_client.py": (
            "import pytest\n\n"
            "from client import Transient, call\n\n\n"
            "def test_retries_until_success():\n"
            "    state = {'n': 0}\n\n"
            "    def flaky():\n"
            "        state['n'] += 1\n"
            "        if state['n'] < 3:\n"
            "            raise Transient('nope')\n"
            "        return 'ok'\n\n"
            "    assert call(flaky, attempts=3) == 'ok'\n"
            "    assert state['n'] == 3\n\n\n"
            "def test_gives_up_after_attempts():\n"
            "    def always():\n"
            "        raise Transient('nope')\n\n"
            "    with pytest.raises(Transient):\n"
            "        call(always, attempts=2)\n\n\n"
            "def test_non_transient_not_retried():\n"
            "    state = {'n': 0}\n\n"
            "    def boom():\n"
            "        state['n'] += 1\n"
            "        raise ValueError('fatal')\n\n"
            "    with pytest.raises(ValueError):\n"
            "        call(boom, attempts=3)\n"
            "    assert state['n'] == 1\n"
        ),
    },
    verifier=("python", "-m", "pytest", "-q", "tests/test_client.py"),
    recorded=RecordedState(
        branch="feat/client-retry",
        task_title="retry transient client failures with backoff",
        remaining_work=(
            "write the retry loop in client.call using delays()",
            "only retry Transient, let other exceptions propagate immediately",
        ),
        decisions=(
            "sleeping is not asserted by the tests; do not add real sleeps that "
            "slow the suite",
        ),
        verifications=(("client retry tests", "failed",
                        "pytest tests/test_client.py"),),
        changed_paths=("client.py",),
        rejected=(
            "catching bare Exception was tried and rejected: it swallows "
            "ValueError and breaks test_non_transient_not_retried",
        ),
    ),
)

# ── S3: repository-wide ────────────────────────────────────────────────

T03 = Task(
    task_id="t03-rename-propagation",
    stratum="repository_wide",
    statement=(
        "The field `qty` was renamed to `quantity` across the order model. "
        "Finish the rename so tests/test_orders.py passes."
    ),
    files={
        "models.py": (
            "class Item:\n"
            "    def __init__(self, sku, quantity):\n"
            "        self.sku = sku\n"
            "        self.quantity = quantity\n"
        ),
        "pricing.py": (
            "def line_total(item, unit_price):\n"
            "    return item.qty * unit_price\n"
        ),
        "report.py": (
            "def summarise(items):\n"
            "    return sum(i.qty for i in items)\n"
        ),
        "tests/test_orders.py": (
            "from models import Item\n"
            "from pricing import line_total\n"
            "from report import summarise\n\n\n"
            "def test_line_total():\n"
            "    assert line_total(Item('a', 3), 250) == 750\n\n\n"
            "def test_summarise():\n"
            "    assert summarise([Item('a', 2), Item('b', 5)]) == 7\n"
        ),
    },
    verifier=("python", "-m", "pytest", "-q", "tests/test_orders.py"),
    recorded=RecordedState(
        branch="refactor/quantity-rename",
        task_title="finish renaming Item.qty to Item.quantity",
        remaining_work=(
            "update pricing.line_total to use item.quantity",
            "update report.summarise to use i.quantity",
        ),
        decisions=(
            "models.py is already migrated and is the source of truth; do not "
            "reintroduce a qty alias",
        ),
        verifications=(("order tests", "failed", "pytest tests/test_orders.py"),),
        changed_paths=("models.py",),
        rejected=(
            "adding a qty property alias on Item was tried and rejected: it "
            "hides remaining call sites instead of migrating them",
        ),
    ),
)

# ── S4: state-heavy ────────────────────────────────────────────────────

T04 = Task(
    task_id="t04-parser-state",
    stratum="state_heavy",
    statement=(
        "tokenizer.py must handle quoted segments containing the delimiter. "
        "Make tests/test_tokenizer.py pass."
    ),
    files={
        "tokenizer.py": (
            "def split_fields(line, delim=','):\n"
            "    # Partial: naive split only; quote handling not implemented.\n"
            "    return line.split(delim)\n"
        ),
        "tests/test_tokenizer.py": (
            "from tokenizer import split_fields\n\n\n"
            "def test_plain():\n"
            "    assert split_fields('a,b,c') == ['a', 'b', 'c']\n\n\n"
            "def test_quoted_delimiter():\n"
            '    assert split_fields(\'a,"b,c",d\') == [\'a\', \'b,c\', \'d\']\n\n\n'
            "def test_empty_fields_preserved():\n"
            "    assert split_fields('a,,b') == ['a', '', 'b']\n\n\n"
            "def test_quotes_stripped_only_when_wrapping():\n"
            '    assert split_fields(\'a,b"c,d\') == [\'a\', \'b"c\', \'d\']\n'
        ),
    },
    verifier=("python", "-m", "pytest", "-q", "tests/test_tokenizer.py"),
    recorded=RecordedState(
        branch="fix/tokenizer-quotes",
        task_title="handle quoted delimiters in split_fields",
        remaining_work=(
            "track in-quote state while scanning instead of splitting",
            "strip quotes only when they wrap the whole field",
        ),
        decisions=(
            "empty fields must survive, so filtering falsy segments is not allowed",
        ),
        verifications=(("tokenizer tests", "failed",
                        "pytest tests/test_tokenizer.py"),),
        changed_paths=("tokenizer.py",),
        rejected=(
            "the csv module was tried and rejected: it also strips a mid-field "
            "quote, which breaks test_quotes_stripped_only_when_wrapping",
        ),
    ),
)

# ── S5: failed-hypothesis-heavy ────────────────────────────────────────

T05 = Task(
    task_id="t05-cache-invalidation",
    stratum="failed_hypothesis",
    statement=(
        "cache.py must invalidate entries when their source version changes. "
        "Make tests/test_cache.py pass."
    ),
    files={
        "cache.py": (
            "class VersionedCache:\n"
            "    def __init__(self):\n"
            "        self._data = {}\n\n"
            "    def put(self, key, value, version):\n"
            "        # Partial: version accepted but not stored.\n"
            "        self._data[key] = value\n\n"
            "    def get(self, key, version):\n"
            "        return self._data.get(key)\n"
        ),
        "tests/test_cache.py": (
            "from cache import VersionedCache\n\n\n"
            "def test_hit_on_same_version():\n"
            "    c = VersionedCache()\n"
            "    c.put('k', 'v1', version=1)\n"
            "    assert c.get('k', version=1) == 'v1'\n\n\n"
            "def test_miss_on_newer_version():\n"
            "    c = VersionedCache()\n"
            "    c.put('k', 'v1', version=1)\n"
            "    assert c.get('k', version=2) is None\n\n\n"
            "def test_miss_on_older_version():\n"
            "    c = VersionedCache()\n"
            "    c.put('k', 'v1', version=5)\n"
            "    assert c.get('k', version=4) is None\n\n\n"
            "def test_reput_replaces_version():\n"
            "    c = VersionedCache()\n"
            "    c.put('k', 'v1', version=1)\n"
            "    c.put('k', 'v2', version=2)\n"
            "    assert c.get('k', version=2) == 'v2'\n"
            "    assert c.get('k', version=1) is None\n"
        ),
    },
    verifier=("python", "-m", "pytest", "-q", "tests/test_cache.py"),
    recorded=RecordedState(
        branch="fix/cache-versioning",
        task_title="invalidate cache entries on source version change",
        remaining_work=(
            "store the version alongside the value in put()",
            "return None from get() unless the stored version equals the "
            "requested version",
        ),
        decisions=(
            "version equality is required in both directions; an older request "
            "must also miss",
        ),
        verifications=(("cache tests", "failed", "pytest tests/test_cache.py"),),
        changed_paths=("cache.py",),
        rejected=(
            "a monotonic 'version >= stored' check was tried and rejected: it "
            "passes test_miss_on_newer_version but fails "
            "test_miss_on_older_version",
            "a TTL-based expiry was tried and rejected: no test involves time, "
            "and it does not key on version at all",
        ),
    ),
)

# ── S6: longer / multi-step ────────────────────────────────────────────

T06 = Task(
    task_id="t06-pipeline-ordering",
    stratum="long_running",
    statement=(
        "pipeline.py must run stages in dependency order and detect cycles. "
        "Make tests/test_pipeline.py pass."
    ),
    files={
        "pipeline.py": (
            "class CycleError(Exception):\n"
            "    pass\n\n\n"
            "def run_order(stages):\n"
            '    """stages: {name: [dependency names]} -> list of names."""\n'
            "    # Partial: signature and CycleError agreed; no sort yet.\n"
            "    return list(stages)\n"
        ),
        "tests/test_pipeline.py": (
            "import pytest\n\n"
            "from pipeline import CycleError, run_order\n\n\n"
            "def test_linear_chain():\n"
            "    order = run_order({'c': ['b'], 'b': ['a'], 'a': []})\n"
            "    assert order.index('a') < order.index('b') < order.index('c')\n\n\n"
            "def test_diamond():\n"
            "    order = run_order({'d': ['b', 'c'], 'b': ['a'], 'c': ['a'], 'a': []})\n"
            "    assert order.index('a') == 0\n"
            "    assert order.index('d') == 3\n\n\n"
            "def test_cycle_detected():\n"
            "    with pytest.raises(CycleError):\n"
            "        run_order({'a': ['b'], 'b': ['a']})\n\n\n"
            "def test_deterministic_for_independent_stages():\n"
            "    a = run_order({'x': [], 'y': [], 'z': []})\n"
            "    b = run_order({'x': [], 'y': [], 'z': []})\n"
            "    assert a == b\n"
        ),
    },
    verifier=("python", "-m", "pytest", "-q", "tests/test_pipeline.py"),
    recorded=RecordedState(
        branch="feat/pipeline-order",
        task_title="topologically order pipeline stages and detect cycles",
        remaining_work=(
            "implement a topological sort over the dependency map",
            "raise CycleError when no stage has its dependencies satisfied",
        ),
        decisions=(
            "independent stages must come out in a stable order, so any set "
            "iteration has to be sorted",
        ),
        verifications=(("pipeline tests", "failed",
                        "pytest tests/test_pipeline.py"),),
        changed_paths=("pipeline.py",),
        rejected=(
            "recursive DFS without a visiting-set was tried and rejected: it "
            "recurses forever on the cycle case instead of raising CycleError",
        ),
    ),
)


SUITE: tuple[Task, ...] = (T01, T02, T03, T04, T05, T06)

__all__ = ["RecordedState", "SUITE", "Task"]
