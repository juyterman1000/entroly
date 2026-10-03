"""Pilot-1: the 24-task continuity suite.

Designed against the failure mode Pilot-0 exposed. In Pilot-0 all three arms
solved task 1, which means that task could not discriminate between having a
handoff and not having one. A suite where B0 does as well as B3 measures task
difficulty, not continuity.

So every task here carries a **trap**: a plausible approach that satisfies some
of the visible tests and fails at least one, which agent A already tried and
recorded as rejected. An agent with no handoff has to discover the trap by
falling into it; an agent with the handoff is told. That is the only mechanism by
which continuation state can pay for itself, and it is what the pilot has to put
under measurement.

Three further constraints, each chosen to keep the comparison honest:

* The trap must be discoverable. If the only way to avoid it were to be told, the
  suite would be rigged for the handoff arms. In every task below the failing
  test does eventually reveal the problem -- the handoff saves the round trip,
  it does not supply unobtainable information.
* `recorded` is the single source for B1 and B3 both. Neither arm can hold a fact
  the other lacks; they differ only in representation.
* The verifier includes the trap case. A solution that only satisfies the obvious
  assertions does not count as verified completion.

Strata are balanced at four tasks each across the six predeclared strata.
"""

from __future__ import annotations

from tasks import RecordedState, Task


def _t(task_id, stratum, statement, files, verifier, recorded):
    return Task(task_id=task_id, stratum=stratum, statement=statement,
                files=files, verifier=verifier, recorded=recorded)


def _pytest(path: str) -> tuple[str, ...]:
    return ("python", "-m", "pytest", "-q", path)


# ══════════════════════════════════════════════════════════════════════
# simple_local (4)
# ══════════════════════════════════════════════════════════════════════

P01 = _t(
    "p01-half-up", "simple_local",
    "money.py must round half-up to 2 places. Make tests/test_money.py pass.",
    {
        "money.py": "from decimal import Decimal\n\n\ndef to_cents(amount):\n    return Decimal(str(amount))\n",
        "tests/test_money.py": (
            "from decimal import Decimal\n\nfrom money import to_cents\n\n\n"
            "def test_half_up():\n"
            "    assert to_cents(2.675) == Decimal('2.68')\n"
            "    assert to_cents(1.005) == Decimal('1.01')\n\n\n"
            "def test_exact_unchanged():\n    assert to_cents(3.10) == Decimal('3.10')\n"
        ),
    },
    _pytest("tests/test_money.py"),
    RecordedState(
        branch="feat/money-rounding",
        task_title="round currency half-up to two places",
        remaining_work=("apply ROUND_HALF_UP quantisation in to_cents",),
        decisions=("amounts stay Decimal end to end; float rounding was the original bug",),
        verifications=(("money tests", "failed", "pytest tests/test_money.py"),),
        changed_paths=("money.py",),
        rejected=("round() was tried and rejected: banker's rounding sends 2.675 to 2.67",),
    ),
)

P02 = _t(
    "p02-slug-collisions", "simple_local",
    "slug.py must produce unique slugs. Make tests/test_slug.py pass.",
    {
        "slug.py": (
            "import re\n\n_seen = set()\n\n\n"
            "def slugify(title):\n"
            "    base = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')\n"
            "    return base\n"
        ),
        "tests/test_slug.py": (
            "from slug import slugify\n\n\n"
            "def test_basic():\n    assert slugify('Hello World') == 'hello-world'\n\n\n"
            "def test_collision_suffix():\n"
            "    assert slugify('Hello World') == 'hello-world'\n"
            "    assert slugify('Hello  World') == 'hello-world-2'\n"
            "    assert slugify('hello world') == 'hello-world-3'\n\n\n"
            "def test_distinct_titles_unaffected():\n"
            "    assert slugify('Other Thing') == 'other-thing'\n"
        ),
    },
    _pytest("tests/test_slug.py"),
    RecordedState(
        branch="feat/slug-uniqueness",
        task_title="suffix duplicate slugs with an incrementing counter",
        remaining_work=("track emitted slugs and append -2, -3 on collision",),
        decisions=("the first occurrence must stay unsuffixed; only later ones get -N",),
        verifications=(("slug tests", "failed", "pytest tests/test_slug.py"),),
        changed_paths=("slug.py",),
        rejected=("hashing the title into the slug was tried and rejected: it changes the first slug too, breaking test_basic",),
    ),
)

P03 = _t(
    "p03-truncate-words", "simple_local",
    "text.py must truncate on word boundaries. Make tests/test_text.py pass.",
    {
        "text.py": "def truncate(s, limit):\n    return s[:limit]\n",
        "tests/test_text.py": (
            "from text import truncate\n\n\n"
            "def test_word_boundary():\n"
            "    assert truncate('hello world foo', 12) == 'hello world'\n\n\n"
            "def test_no_truncation_needed():\n"
            "    assert truncate('short', 12) == 'short'\n\n\n"
            "def test_single_long_word_is_hard_cut():\n"
            "    assert truncate('abcdefghijklmno', 5) == 'abcde'\n"
        ),
    },
    _pytest("tests/test_text.py"),
    RecordedState(
        branch="fix/truncate-words",
        task_title="truncate text on word boundaries",
        remaining_work=("cut at the last space within the limit", "fall back to a hard cut when there is no space"),
        decisions=("no ellipsis is added; the tests compare exact strings",),
        verifications=(("text tests", "failed", "pytest tests/test_text.py"),),
        changed_paths=("text.py",),
        rejected=("rsplit(' ', 1)[0] was tried and rejected: with no space in the string it returns the whole word, failing the hard-cut case",),
    ),
)

P04 = _t(
    "p04-percent-change", "simple_local",
    "metrics.py must compute percent change safely. Make tests/test_metrics.py pass.",
    {
        "metrics.py": "def pct_change(old, new):\n    return (new - old) / old * 100\n",
        "tests/test_metrics.py": (
            "from metrics import pct_change\n\n\n"
            "def test_normal():\n    assert pct_change(200, 250) == 25.0\n\n\n"
            "def test_from_zero_is_none():\n    assert pct_change(0, 5) is None\n\n\n"
            "def test_zero_to_zero_is_zero():\n    assert pct_change(0, 0) == 0.0\n\n\n"
            "def test_negative_baseline():\n    assert pct_change(-100, -50) == -50.0\n"
        ),
    },
    _pytest("tests/test_metrics.py"),
    RecordedState(
        branch="fix/pct-change-guards",
        task_title="guard percent change against a zero baseline",
        remaining_work=("return None when old is 0 and new is not", "return 0.0 when both are 0"),
        decisions=("a negative baseline is a real case and must keep the signed formula",),
        verifications=(("metrics tests", "failed", "pytest tests/test_metrics.py"),),
        changed_paths=("metrics.py",),
        rejected=("using abs(old) in the denominator was tried and rejected: it makes the negative-baseline case +50 instead of -50",),
    ),
)

# ══════════════════════════════════════════════════════════════════════
# multi_file (4)
# ══════════════════════════════════════════════════════════════════════

P05 = _t(
    "p05-retry-transient", "multi_file",
    "client.py should retry transient failures using backoff.py. Make tests/test_client.py pass.",
    {
        "backoff.py": "def delays(attempts, base=0.1):\n    return [base * (2 ** i) for i in range(attempts)]\n",
        "client.py": (
            "from backoff import delays\n\n\nclass Transient(Exception):\n    pass\n\n\n"
            "def call(fn, attempts=3):\n    return fn()\n"
        ),
        "tests/test_client.py": (
            "import pytest\n\nfrom client import Transient, call\n\n\n"
            "def test_retries_until_success():\n"
            "    state = {'n': 0}\n\n    def flaky():\n        state['n'] += 1\n"
            "        if state['n'] < 3:\n            raise Transient('nope')\n        return 'ok'\n\n"
            "    assert call(flaky, attempts=3) == 'ok'\n    assert state['n'] == 3\n\n\n"
            "def test_gives_up():\n"
            "    def always():\n        raise Transient('nope')\n\n"
            "    with pytest.raises(Transient):\n        call(always, attempts=2)\n\n\n"
            "def test_non_transient_not_retried():\n"
            "    state = {'n': 0}\n\n    def boom():\n        state['n'] += 1\n        raise ValueError('fatal')\n\n"
            "    with pytest.raises(ValueError):\n        call(boom, attempts=3)\n    assert state['n'] == 1\n"
        ),
    },
    _pytest("tests/test_client.py"),
    RecordedState(
        branch="feat/client-retry",
        task_title="retry transient client failures with backoff",
        remaining_work=("write the retry loop in call() using delays()", "retry only Transient"),
        decisions=("no real sleeps; the tests do not assert timing and sleeping slows the suite",),
        verifications=(("client tests", "failed", "pytest tests/test_client.py"),),
        changed_paths=("client.py",),
        rejected=("catching bare Exception was tried and rejected: it swallows ValueError and retries it three times",),
    ),
)

P06 = _t(
    "p06-config-layering", "multi_file",
    "config.py must layer defaults, file and env. Make tests/test_config.py pass.",
    {
        "defaults.py": "DEFAULTS = {'host': 'localhost', 'port': 8080, 'debug': False}\n",
        "config.py": (
            "import os\n\nfrom defaults import DEFAULTS\n\n\n"
            "def load(file_values=None):\n"
            "    merged = dict(DEFAULTS)\n"
            "    merged.update(file_values or {})\n"
            "    return merged\n"
        ),
        "tests/test_config.py": (
            "import os\n\nimport pytest\n\nfrom config import load\n\n\n"
            "@pytest.fixture(autouse=True)\n"
            "def clean_env(monkeypatch):\n"
            "    for key in ('APP_HOST', 'APP_PORT', 'APP_DEBUG'):\n"
            "        monkeypatch.delenv(key, raising=False)\n\n\n"
            "def test_defaults():\n    assert load()['port'] == 8080\n\n\n"
            "def test_file_overrides_default():\n"
            "    assert load({'port': 9000})['port'] == 9000\n\n\n"
            "def test_env_overrides_file(monkeypatch):\n"
            "    monkeypatch.setenv('APP_PORT', '7000')\n"
            "    assert load({'port': 9000})['port'] == 7000\n\n\n"
            "def test_env_port_is_int(monkeypatch):\n"
            "    monkeypatch.setenv('APP_PORT', '7000')\n"
            "    assert isinstance(load()['port'], int)\n\n\n"
            "def test_env_debug_is_bool(monkeypatch):\n"
            "    monkeypatch.setenv('APP_DEBUG', 'false')\n"
            "    assert load()['debug'] is False\n"
        ),
    },
    _pytest("tests/test_config.py"),
    RecordedState(
        branch="feat/config-layering",
        task_title="layer config as defaults then file then environment",
        remaining_work=("apply APP_* environment overrides last", "coerce env values to the type of the default"),
        decisions=("precedence is defaults < file < env and must not be reordered",),
        verifications=(("config tests", "failed", "pytest tests/test_config.py"),),
        changed_paths=("config.py",),
        rejected=("passing env strings straight through was tried and rejected: port becomes '7000' and APP_DEBUG='false' is truthy, so two tests fail",),
    ),
)

P07 = _t(
    "p07-event-dispatch", "multi_file",
    "bus.py must dispatch to handlers in registration order. Make tests/test_bus.py pass.",
    {
        "handlers.py": (
            "calls = []\n\n\ndef first(event):\n    calls.append(('first', event))\n\n\n"
            "def second(event):\n    calls.append(('second', event))\n\n\n"
            "def failing(event):\n    raise RuntimeError('handler blew up')\n"
        ),
        "bus.py": (
            "class Bus:\n"
            "    def __init__(self):\n        self._handlers = {}\n\n"
            "    def on(self, name, fn):\n        self._handlers.setdefault(name, []).append(fn)\n\n"
            "    def emit(self, name, event):\n"
            "        for fn in self._handlers.get(name, []):\n            fn(event)\n"
        ),
    },
    _pytest("tests/test_bus.py"),
    RecordedState(
        branch="feat/bus-isolation",
        task_title="isolate handler failures without dropping later handlers",
        remaining_work=("catch per-handler exceptions in emit so later handlers still run", "return the list of exceptions raised"),
        decisions=("registration order is part of the contract; do not sort or reverse handlers",),
        verifications=(("bus tests", "failed", "pytest tests/test_bus.py"),),
        changed_paths=("bus.py",),
        rejected=("wrapping the whole loop in one try/except was tried and rejected: the first failure aborts the remaining handlers",),
    ),
)
P07.files["tests/test_bus.py"] = (
    "import handlers\nfrom bus import Bus\n\n\n"
    "def setup_function():\n    handlers.calls.clear()\n\n\n"
    "def test_order_preserved():\n"
    "    bus = Bus()\n    bus.on('x', handlers.first)\n    bus.on('x', handlers.second)\n"
    "    bus.emit('x', 1)\n"
    "    assert [name for name, _ in handlers.calls] == ['first', 'second']\n\n\n"
    "def test_failure_does_not_stop_later_handlers():\n"
    "    bus = Bus()\n    bus.on('x', handlers.failing)\n    bus.on('x', handlers.second)\n"
    "    errors = bus.emit('x', 1)\n"
    "    assert [name for name, _ in handlers.calls] == ['second']\n"
    "    assert len(errors) == 1\n\n\n"
    "def test_no_handlers_returns_empty():\n"
    "    assert Bus().emit('missing', 1) == []\n"
)

P08 = _t(
    "p08-path-normalise", "multi_file",
    "paths.py must normalise repo-relative paths. Make tests/test_paths.py pass.",
    {
        "paths.py": (
            "def normalise(path):\n"
            "    return path.replace('\\\\', '/')\n"
        ),
        "tests/test_paths.py": (
            "import pytest\n\nfrom paths import normalise\n\n\n"
            "def test_windows_separators():\n    assert normalise('a\\\\b\\\\c.py') == 'a/b/c.py'\n\n\n"
            "def test_leading_dot_removed():\n    assert normalise('./a/b.py') == 'a/b.py'\n\n\n"
            "def test_collapses_double_slash():\n    assert normalise('a//b.py') == 'a/b.py'\n\n\n"
            "def test_escape_rejected():\n"
            "    with pytest.raises(ValueError):\n        normalise('../secrets.py')\n"
        ),
    },
    _pytest("tests/test_paths.py"),
    RecordedState(
        branch="fix/path-normalise",
        task_title="normalise repo-relative paths and reject escapes",
        remaining_work=("strip a leading ./ and collapse repeated slashes", "raise ValueError on any .. segment"),
        decisions=("paths stay relative; never return an absolute path",),
        verifications=(("path tests", "failed", "pytest tests/test_paths.py"),),
        changed_paths=("paths.py",),
        rejected=("os.path.normpath was tried and rejected: it resolves ../secrets.py silently instead of raising, and on Windows it returns backslashes again",),
    ),
)

# ══════════════════════════════════════════════════════════════════════
# repository_wide (4)
# ══════════════════════════════════════════════════════════════════════

P09 = _t(
    "p09-quantity-rename", "repository_wide",
    "Item.qty was renamed to Item.quantity. Finish the rename so tests/test_orders.py passes.",
    {
        "models.py": "class Item:\n    def __init__(self, sku, quantity):\n        self.sku = sku\n        self.quantity = quantity\n",
        "pricing.py": "def line_total(item, unit_price):\n    return item.qty * unit_price\n",
        "report.py": "def summarise(items):\n    return sum(i.qty for i in items)\n",
        "export.py": "def rows(items):\n    return [(i.sku, i.qty) for i in items]\n",
        "tests/test_orders.py": (
            "from export import rows\nfrom models import Item\nfrom pricing import line_total\nfrom report import summarise\n\n\n"
            "def test_line_total():\n    assert line_total(Item('a', 3), 250) == 750\n\n\n"
            "def test_summarise():\n    assert summarise([Item('a', 2), Item('b', 5)]) == 7\n\n\n"
            "def test_rows():\n    assert rows([Item('a', 2)]) == [('a', 2)]\n\n\n"
            "def test_no_qty_attribute():\n"
            "    assert not hasattr(Item('a', 1), 'qty')\n"
        ),
    },
    _pytest("tests/test_orders.py"),
    RecordedState(
        branch="refactor/quantity-rename",
        task_title="finish renaming Item.qty to Item.quantity",
        remaining_work=("update pricing.line_total", "update report.summarise", "update export.rows"),
        decisions=("models.py is migrated and is the source of truth",),
        verifications=(("order tests", "failed", "pytest tests/test_orders.py"),),
        changed_paths=("models.py",),
        rejected=("adding a qty property alias on Item was tried and rejected: test_no_qty_attribute asserts the alias is gone",),
    ),
)

P10 = _t(
    "p10-error-codes", "repository_wide",
    "All API errors must carry a stable code. Make tests/test_errors.py pass.",
    {
        "errors.py": (
            "class AppError(Exception):\n"
            "    code = 'app_error'\n\n"
            "    def __init__(self, message):\n        super().__init__(message)\n        self.message = message\n"
        ),
        "auth.py": "class Unauthorized(Exception):\n    pass\n",
        "billing.py": "class CardDeclined(Exception):\n    pass\n",
        "tests/test_errors.py": (
            "from auth import Unauthorized\nfrom billing import CardDeclined\nfrom errors import AppError\n\n\n"
            "def test_base_code():\n    assert AppError('x').code == 'app_error'\n\n\n"
            "def test_unauthorized_code():\n"
            "    err = Unauthorized('no token')\n"
            "    assert isinstance(err, AppError)\n    assert err.code == 'unauthorized'\n\n\n"
            "def test_card_declined_code():\n"
            "    err = CardDeclined('declined')\n"
            "    assert isinstance(err, AppError)\n    assert err.code == 'card_declined'\n\n\n"
            "def test_message_preserved():\n    assert CardDeclined('declined').message == 'declined'\n"
        ),
    },
    _pytest("tests/test_errors.py"),
    RecordedState(
        branch="feat/stable-error-codes",
        task_title="give every API error a stable code",
        remaining_work=("make Unauthorized and CardDeclined subclass AppError", "set code on each subclass"),
        decisions=("codes are snake_case and must stay stable; they are part of the public API",),
        verifications=(("error tests", "failed", "pytest tests/test_errors.py"),),
        changed_paths=("errors.py",),
        rejected=("deriving the code from the class name at runtime was tried and rejected: it produces 'carddeclined', not 'card_declined'",),
    ),
)

P11 = _t(
    "p11-import-cycle", "repository_wide",
    "Break the import cycle so tests/test_graph.py passes.",
    {
        "node.py": "from edge import Edge\n\n\nclass Node:\n    def __init__(self, name):\n        self.name = name\n\n    def to(self, other):\n        return Edge(self, other)\n",
        "edge.py": "from node import Node\n\n\nclass Edge:\n    def __init__(self, a, b):\n        self.a = a\n        self.b = b\n\n    def endpoints(self):\n        assert isinstance(self.a, Node)\n        return (self.a.name, self.b.name)\n",
        "tests/test_graph.py": (
            "def test_import_node_first():\n"
            "    from node import Node\n"
            "    assert Node('a').to(Node('b')).endpoints() == ('a', 'b')\n\n\n"
            "def test_import_edge_first():\n"
            "    import importlib\n    import sys\n"
            "    for name in ('node', 'edge'):\n        sys.modules.pop(name, None)\n"
            "    edge = importlib.import_module('edge')\n"
            "    node = importlib.import_module('node')\n"
            "    assert edge.Edge(node.Node('a'), node.Node('b')).endpoints() == ('a', 'b')\n"
        ),
    },
    _pytest("tests/test_graph.py"),
    RecordedState(
        branch="refactor/break-cycle",
        task_title="break the node/edge import cycle",
        remaining_work=("remove one module-level import and keep the isinstance check working",),
        decisions=("both import orders must work; the test exercises each",),
        verifications=(("graph tests", "failed", "pytest tests/test_graph.py"),),
        changed_paths=("node.py", "edge.py"),
        rejected=("deleting the isinstance check was tried and rejected: it removes the assertion the cycle existed to support instead of fixing the cycle",),
    ),
)

P12 = _t(
    "p12-settings-single-source", "repository_wide",
    "Timeout must come from one place. Make tests/test_timeout.py pass.",
    {
        "settings.py": "TIMEOUT_SECONDS = 30\n",
        "http.py": "TIMEOUT = 10\n\n\ndef fetch(url):\n    return ('GET', url, TIMEOUT)\n",
        "worker.py": "def run(job):\n    return ('job', job, 60)\n",
        "tests/test_timeout.py": (
            "import http as app_http\nimport settings\nimport worker\n\n\n"
            "def test_http_uses_settings():\n    assert app_http.fetch('u')[2] == settings.TIMEOUT_SECONDS\n\n\n"
            "def test_worker_uses_settings():\n    assert worker.run('j')[2] == settings.TIMEOUT_SECONDS\n\n\n"
            "def test_override_propagates(monkeypatch):\n"
            "    monkeypatch.setattr(settings, 'TIMEOUT_SECONDS', 5)\n"
            "    assert app_http.fetch('u')[2] == 5\n    assert worker.run('j')[2] == 5\n"
        ),
    },
    _pytest("tests/test_timeout.py"),
    RecordedState(
        branch="refactor/one-timeout",
        task_title="read the timeout from settings everywhere",
        remaining_work=("remove http.TIMEOUT and read settings at call time", "remove worker's hardcoded 60"),
        decisions=("settings.TIMEOUT_SECONDS is the single source of truth",),
        verifications=(("timeout tests", "failed", "pytest tests/test_timeout.py"),),
        changed_paths=("settings.py",),
        rejected=("from settings import TIMEOUT_SECONDS at module import was tried and rejected: it binds the value once, so the monkeypatch override test still fails",),
    ),
)

# ══════════════════════════════════════════════════════════════════════
# state_heavy (4)
# ══════════════════════════════════════════════════════════════════════

P13 = _t(
    "p13-quoted-delimiter", "state_heavy",
    "tokenizer.py must handle quoted segments containing the delimiter. Make tests/test_tokenizer.py pass.",
    {
        "tokenizer.py": "def split_fields(line, delim=','):\n    return line.split(delim)\n",
        "tests/test_tokenizer.py": (
            "from tokenizer import split_fields\n\n\n"
            "def test_plain():\n    assert split_fields('a,b,c') == ['a', 'b', 'c']\n\n\n"
            "def test_quoted_delimiter():\n"
            "    assert split_fields('a,\"b,c\",d') == ['a', 'b,c', 'd']\n\n\n"
            "def test_empty_fields_preserved():\n    assert split_fields('a,,b') == ['a', '', 'b']\n\n\n"
            "def test_midfield_quote_kept():\n"
            "    assert split_fields('a,b\"c,d') == ['a', 'b\"c', 'd']\n"
        ),
    },
    _pytest("tests/test_tokenizer.py"),
    RecordedState(
        branch="fix/tokenizer-quotes",
        task_title="handle quoted delimiters in split_fields",
        remaining_work=("track in-quote state while scanning", "strip quotes only when they wrap the whole field"),
        decisions=("empty fields must survive, so filtering falsy segments is not allowed",),
        verifications=(("tokenizer tests", "failed", "pytest tests/test_tokenizer.py"),),
        changed_paths=("tokenizer.py",),
        rejected=("the csv module was tried and rejected: it also strips the mid-field quote, failing test_midfield_quote_kept",),
    ),
)

P14 = _t(
    "p14-version-cache", "state_heavy",
    "cache.py must invalidate on source version change. Make tests/test_cache.py pass.",
    {
        "cache.py": (
            "class VersionedCache:\n"
            "    def __init__(self):\n        self._data = {}\n\n"
            "    def put(self, key, value, version):\n        self._data[key] = value\n\n"
            "    def get(self, key, version):\n        return self._data.get(key)\n"
        ),
        "tests/test_cache.py": (
            "from cache import VersionedCache\n\n\n"
            "def test_hit_same_version():\n"
            "    c = VersionedCache()\n    c.put('k', 'v1', version=1)\n"
            "    assert c.get('k', version=1) == 'v1'\n\n\n"
            "def test_miss_newer():\n"
            "    c = VersionedCache()\n    c.put('k', 'v1', version=1)\n"
            "    assert c.get('k', version=2) is None\n\n\n"
            "def test_miss_older():\n"
            "    c = VersionedCache()\n    c.put('k', 'v1', version=5)\n"
            "    assert c.get('k', version=4) is None\n\n\n"
            "def test_reput_replaces():\n"
            "    c = VersionedCache()\n    c.put('k', 'v1', version=1)\n    c.put('k', 'v2', version=2)\n"
            "    assert c.get('k', version=2) == 'v2'\n    assert c.get('k', version=1) is None\n"
        ),
    },
    _pytest("tests/test_cache.py"),
    RecordedState(
        branch="fix/cache-versioning",
        task_title="invalidate cache entries on version change",
        remaining_work=("store the version with the value", "miss unless the stored version equals the requested one"),
        decisions=("equality in both directions; an older request must also miss",),
        verifications=(("cache tests", "failed", "pytest tests/test_cache.py"),),
        changed_paths=("cache.py",),
        rejected=("a monotonic version >= stored check was tried and rejected: it passes the newer case and fails test_miss_older",),
    ),
)

P15 = _t(
    "p15-session-expiry", "state_heavy",
    "session.py must expire idle sessions. Make tests/test_session.py pass.",
    {
        "session.py": (
            "class Sessions:\n"
            "    def __init__(self, idle_ttl=10):\n        self._idle_ttl = idle_ttl\n        self._store = {}\n\n"
            "    def touch(self, sid, now):\n        self._store[sid] = now\n\n"
            "    def active(self, sid, now):\n        return sid in self._store\n"
        ),
        "tests/test_session.py": (
            "from session import Sessions\n\n\n"
            "def test_active_within_ttl():\n"
            "    s = Sessions(idle_ttl=10)\n    s.touch('a', now=0)\n"
            "    assert s.active('a', now=5) is True\n\n\n"
            "def test_expired_after_ttl():\n"
            "    s = Sessions(idle_ttl=10)\n    s.touch('a', now=0)\n"
            "    assert s.active('a', now=11) is False\n\n\n"
            "def test_touch_extends():\n"
            "    s = Sessions(idle_ttl=10)\n    s.touch('a', now=0)\n    s.touch('a', now=8)\n"
            "    assert s.active('a', now=15) is True\n\n\n"
            "def test_expired_session_is_evicted():\n"
            "    s = Sessions(idle_ttl=10)\n    s.touch('a', now=0)\n    s.active('a', now=11)\n"
            "    assert len(s._store) == 0\n"
        ),
    },
    _pytest("tests/test_session.py"),
    RecordedState(
        branch="feat/session-expiry",
        task_title="expire idle sessions and evict them",
        remaining_work=("compare now against the last touch plus idle_ttl", "delete the entry when it is found expired"),
        decisions=("expiry is idle-based, measured from the last touch, not from creation",),
        verifications=(("session tests", "failed", "pytest tests/test_session.py"),),
        changed_paths=("session.py",),
        rejected=("a read-only active() check was tried and rejected: it returns the right booleans but leaves the entry in _store, failing test_expired_session_is_evicted",),
    ),
)

P16 = _t(
    "p16-dedupe-stream", "state_heavy",
    "stream.py must drop repeats inside a sliding window. Make tests/test_stream.py pass.",
    {
        "stream.py": (
            "def dedupe(items, window=3):\n"
            "    seen = set()\n    out = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n            seen.add(item)\n            out.append(item)\n"
            "    return out\n"
        ),
        "tests/test_stream.py": (
            "from stream import dedupe\n\n\n"
            "def test_adjacent_repeat_dropped():\n"
            "    assert dedupe(['a', 'a', 'b'], window=3) == ['a', 'b']\n\n\n"
            "def test_repeat_outside_window_kept():\n"
            "    assert dedupe(['a', 'b', 'c', 'd', 'a'], window=3) == ['a', 'b', 'c', 'd', 'a']\n\n\n"
            "def test_repeat_inside_window_dropped():\n"
            "    assert dedupe(['a', 'b', 'a'], window=3) == ['a', 'b']\n\n\n"
            "def test_order_preserved():\n"
            "    assert dedupe(['c', 'b', 'a'], window=3) == ['c', 'b', 'a']\n"
        ),
    },
    _pytest("tests/test_stream.py"),
    RecordedState(
        branch="fix/sliding-dedupe",
        task_title="dedupe only within a sliding window",
        remaining_work=("bound the seen set to the last `window` emitted items",),
        decisions=("the window counts emitted items, not consumed ones",),
        verifications=(("stream tests", "failed", "pytest tests/test_stream.py"),),
        changed_paths=("stream.py",),
        rejected=("an unbounded seen set is the current code and is rejected: it drops the 'a' that reappears outside the window, failing test_repeat_outside_window_kept",),
    ),
)

# ══════════════════════════════════════════════════════════════════════
# failed_hypothesis (4)
# ══════════════════════════════════════════════════════════════════════

P17 = _t(
    "p17-rate-limit-window", "failed_hypothesis",
    "limiter.py must allow N requests per sliding window. Make tests/test_limiter.py pass.",
    {
        "limiter.py": (
            "class Limiter:\n"
            "    def __init__(self, limit=2, window=10):\n"
            "        self.limit = limit\n        self.window = window\n        self._hits = {}\n\n"
            "    def allow(self, key, now):\n"
            "        self._hits.setdefault(key, 0)\n"
            "        self._hits[key] += 1\n"
            "        return self._hits[key] <= self.limit\n"
        ),
        "tests/test_limiter.py": (
            "from limiter import Limiter\n\n\n"
            "def test_under_limit():\n"
            "    lim = Limiter(limit=2, window=10)\n"
            "    assert lim.allow('k', now=0) is True\n    assert lim.allow('k', now=1) is True\n\n\n"
            "def test_over_limit():\n"
            "    lim = Limiter(limit=2, window=10)\n"
            "    lim.allow('k', now=0)\n    lim.allow('k', now=1)\n"
            "    assert lim.allow('k', now=2) is False\n\n\n"
            "def test_window_slides():\n"
            "    lim = Limiter(limit=2, window=10)\n"
            "    lim.allow('k', now=0)\n    lim.allow('k', now=1)\n"
            "    assert lim.allow('k', now=12) is True\n\n\n"
            "def test_keys_independent():\n"
            "    lim = Limiter(limit=1, window=10)\n"
            "    assert lim.allow('a', now=0) is True\n    assert lim.allow('b', now=0) is True\n"
        ),
    },
    _pytest("tests/test_limiter.py"),
    RecordedState(
        branch="fix/sliding-rate-limit",
        task_title="make the rate limiter window actually slide",
        remaining_work=("store per-key timestamps and drop ones older than the window", "count only the surviving timestamps"),
        decisions=("keys are independent and must never share a counter",),
        verifications=(("limiter tests", "failed", "pytest tests/test_limiter.py"),),
        changed_paths=("limiter.py",),
        rejected=(
            "a plain counter is the current code and is rejected: it never forgets, so test_window_slides fails",
            "resetting the counter when now exceeds the window was tried and rejected: it is a fixed window, so two requests at now=9 and two at now=11 both pass and the limit is effectively doubled",
        ),
    ),
)

P18 = _t(
    "p18-merge-conflict-markers", "failed_hypothesis",
    "patchcheck.py must reject conflict markers. Make tests/test_patchcheck.py pass.",
    {
        "patchcheck.py": (
            "MARKERS = ('<<<<<<<', '=======', '>>>>>>>')\n\n\n"
            "def has_conflict(text):\n"
            "    return any(m in text for m in MARKERS)\n"
        ),
        "tests/test_patchcheck.py": (
            "from patchcheck import has_conflict\n\n\n"
            "def test_real_conflict():\n"
            "    assert has_conflict('a\\n<<<<<<< HEAD\\nb\\n=======\\nc\\n>>>>>>> x\\n') is True\n\n\n"
            "def test_clean_text():\n    assert has_conflict('just code\\n') is False\n\n\n"
            "def test_markdown_rule_is_not_a_conflict():\n"
            "    assert has_conflict('Title\\n=======\\n\\nbody\\n') is False\n\n\n"
            "def test_marker_must_start_line():\n"
            "    assert has_conflict('x <<<<<<< inline\\n') is False\n"
        ),
    },
    _pytest("tests/test_patchcheck.py"),
    RecordedState(
        branch="fix/conflict-detection",
        task_title="detect real conflict markers without false positives",
        remaining_work=("require a start marker at line start", "require the matching separator and end marker to follow it"),
        decisions=("a markdown setext underline is legal content and must not be flagged",),
        verifications=(("patchcheck tests", "failed", "pytest tests/test_patchcheck.py"),),
        changed_paths=("patchcheck.py",),
        rejected=(
            "substring matching is the current code and is rejected: '=======' alone flags markdown headings",
            "anchoring each marker to line start independently was tried and rejected: the markdown case still trips on its own line-initial '=======', so the markers have to be checked as a sequence",
        ),
    ),
)

P19 = _t(
    "p19-partial-sum-overflow", "failed_hypothesis",
    "stats.py must compute a numerically stable mean. Make tests/test_stats.py pass.",
    {
        "stats.py": "def mean(values):\n    return sum(values) / len(values)\n",
        "tests/test_stats.py": (
            "import pytest\n\nfrom stats import mean\n\n\n"
            "def test_simple():\n    assert mean([1, 2, 3]) == 2\n\n\n"
            "def test_empty_is_none():\n    assert mean([]) is None\n\n\n"
            "def test_large_offset_precision():\n"
            "    values = [1e16, 1.0, 1.0, 1.0, 1.0]\n"
            "    assert mean(values) == pytest.approx(2e15, rel=1e-9)\n"
            "    assert mean(values) != 2000000000000000.0 - 1\n\n\n"
            "def test_mixed_signs():\n    assert mean([-1, 1]) == 0\n"
        ),
    },
    _pytest("tests/test_stats.py"),
    RecordedState(
        branch="fix/stable-mean",
        task_title="numerically stable mean with an empty guard",
        remaining_work=("return None for an empty sequence", "use a compensated or incremental sum so large offsets do not lose the small terms"),
        decisions=("the result must stay a float; do not switch the public API to Decimal",),
        verifications=(("stats tests", "failed", "pytest tests/test_stats.py"),),
        changed_paths=("stats.py",),
        rejected=("sorting the values before summing was tried and rejected: it reduces but does not eliminate the error, and it changes the cost from linear to n log n for no test benefit",),
    ),
)

P20 = _t(
    "p20-idempotent-migration", "failed_hypothesis",
    "migrate.py must apply each migration exactly once. Make tests/test_migrate.py pass.",
    {
        "migrate.py": (
            "def apply(migrations, applied):\n"
            "    log = []\n"
            "    for name, fn in migrations:\n"
            "        fn()\n        log.append(name)\n"
            "    return log\n"
        ),
        "tests/test_migrate.py": (
            "import pytest\n\nfrom migrate import apply\n\n\n"
            "def _m(calls, name):\n    return (name, lambda: calls.append(name))\n\n\n"
            "def test_applies_pending():\n"
            "    calls = []\n"
            "    assert apply([_m(calls, 'a'), _m(calls, 'b')], applied=set()) == ['a', 'b']\n"
            "    assert calls == ['a', 'b']\n\n\n"
            "def test_skips_applied():\n"
            "    calls = []\n"
            "    assert apply([_m(calls, 'a'), _m(calls, 'b')], applied={'a'}) == ['b']\n"
            "    assert calls == ['b']\n\n\n"
            "def test_stops_on_failure():\n"
            "    calls = []\n"
            "    def boom():\n        raise RuntimeError('bad migration')\n"
            "    with pytest.raises(RuntimeError):\n"
            "        apply([_m(calls, 'a'), ('b', boom), _m(calls, 'c')], applied=set())\n"
            "    assert calls == ['a']\n\n\n"
            "def test_applied_set_updated():\n"
            "    calls = []\n    applied = set()\n"
            "    apply([_m(calls, 'a')], applied=applied)\n"
            "    assert applied == {'a'}\n"
        ),
    },
    _pytest("tests/test_migrate.py"),
    RecordedState(
        branch="fix/idempotent-migrations",
        task_title="apply each migration exactly once and stop on failure",
        remaining_work=("skip migrations already in `applied`", "record each success into `applied`", "let an exception propagate without running later migrations"),
        decisions=("`applied` is mutated in place; callers rely on seeing the update",),
        verifications=(("migrate tests", "failed", "pytest tests/test_migrate.py"),),
        changed_paths=("migrate.py",),
        rejected=(
            "wrapping each migration in try/except and continuing was tried and rejected: test_stops_on_failure requires 'c' not to run",
            "returning a new applied set instead of mutating was tried and rejected: test_applied_set_updated inspects the caller's set",
        ),
    ),
)

# ══════════════════════════════════════════════════════════════════════
# long_running (4)
# ══════════════════════════════════════════════════════════════════════

P21 = _t(
    "p21-topological-order", "long_running",
    "pipeline.py must order stages by dependency and detect cycles. Make tests/test_pipeline.py pass.",
    {
        "pipeline.py": (
            "class CycleError(Exception):\n    pass\n\n\n"
            "def run_order(stages):\n    return list(stages)\n"
        ),
        "tests/test_pipeline.py": (
            "import pytest\n\nfrom pipeline import CycleError, run_order\n\n\n"
            "def test_linear():\n"
            "    order = run_order({'c': ['b'], 'b': ['a'], 'a': []})\n"
            "    assert order.index('a') < order.index('b') < order.index('c')\n\n\n"
            "def test_diamond():\n"
            "    order = run_order({'d': ['b', 'c'], 'b': ['a'], 'c': ['a'], 'a': []})\n"
            "    assert order.index('a') == 0 and order.index('d') == 3\n\n\n"
            "def test_cycle():\n"
            "    with pytest.raises(CycleError):\n        run_order({'a': ['b'], 'b': ['a']})\n\n\n"
            "def test_deterministic():\n"
            "    assert run_order({'x': [], 'y': [], 'z': []}) == run_order({'x': [], 'y': [], 'z': []})\n"
        ),
    },
    _pytest("tests/test_pipeline.py"),
    RecordedState(
        branch="feat/pipeline-order",
        task_title="topologically order pipeline stages and detect cycles",
        remaining_work=("implement a topological sort", "raise CycleError when no stage is ready"),
        decisions=("independent stages must come out in a stable order, so any set iteration has to be sorted",),
        verifications=(("pipeline tests", "failed", "pytest tests/test_pipeline.py"),),
        changed_paths=("pipeline.py",),
        rejected=("recursive DFS without a visiting set was tried and rejected: it recurses forever on the cycle case instead of raising",),
    ),
)

P22 = _t(
    "p22-chunked-upload", "long_running",
    "upload.py must resume partial uploads. Make tests/test_upload.py pass.",
    {
        "upload.py": (
            "def upload(data, chunk_size, already_sent=0, sink=None):\n"
            "    sink = [] if sink is None else sink\n"
            "    for i in range(0, len(data), chunk_size):\n"
            "        sink.append(data[i:i + chunk_size])\n"
            "    return sink\n"
        ),
        "tests/test_upload.py": (
            "from upload import upload\n\n\n"
            "def test_full_upload():\n"
            "    assert upload('abcdef', 2) == ['ab', 'cd', 'ef']\n\n\n"
            "def test_resume_skips_sent_bytes():\n"
            "    assert upload('abcdef', 2, already_sent=4) == ['ef']\n\n\n"
            "def test_resume_mid_chunk():\n"
            "    assert upload('abcdef', 2, already_sent=3) == ['de', 'f']\n\n\n"
            "def test_nothing_left():\n"
            "    assert upload('abcdef', 2, already_sent=6) == []\n"
        ),
    },
    _pytest("tests/test_upload.py"),
    RecordedState(
        branch="feat/resumable-upload",
        task_title="resume uploads from a byte offset",
        remaining_work=("start chunking at already_sent", "re-chunk from the offset so a mid-chunk resume realigns"),
        decisions=("chunk boundaries are recomputed from the offset, not from the original alignment",),
        verifications=(("upload tests", "failed", "pytest tests/test_upload.py"),),
        changed_paths=("upload.py",),
        rejected=("skipping already_sent // chunk_size whole chunks was tried and rejected: with already_sent=3 it drops byte 'd', failing test_resume_mid_chunk",),
    ),
)

P23 = _t(
    "p23-backfill-batches", "long_running",
    "backfill.py must process records in bounded batches with progress. Make tests/test_backfill.py pass.",
    {
        "backfill.py": (
            "def run(records, batch_size, on_batch=None):\n"
            "    for record in records:\n"
            "        pass\n"
            "    return {'processed': 0, 'batches': 0}\n"
        ),
        "tests/test_backfill.py": (
            "from backfill import run\n\n\n"
            "def test_counts():\n"
            "    assert run(range(10), batch_size=4) == {'processed': 10, 'batches': 3}\n\n\n"
            "def test_batch_callback_sizes():\n"
            "    sizes = []\n"
            "    run(range(10), batch_size=4, on_batch=lambda b: sizes.append(len(b)))\n"
            "    assert sizes == [4, 4, 2]\n\n\n"
            "def test_empty():\n"
            "    assert run([], batch_size=4) == {'processed': 0, 'batches': 0}\n\n\n"
            "def test_generator_consumed_once():\n"
            "    gen = (i for i in range(5))\n"
            "    assert run(gen, batch_size=2)['processed'] == 5\n"
        ),
    },
    _pytest("tests/test_backfill.py"),
    RecordedState(
        branch="feat/bounded-backfill",
        task_title="process backfill records in bounded batches",
        remaining_work=("accumulate records into batches of batch_size", "invoke on_batch per batch including the short final one", "return processed and batch counts"),
        decisions=("the input may be a one-shot generator, so it must be iterated exactly once",),
        verifications=(("backfill tests", "failed", "pytest tests/test_backfill.py"),),
        changed_paths=("backfill.py",),
        rejected=("len(records) and slicing was tried and rejected: a generator has no length and would be consumed by the first pass, failing test_generator_consumed_once",),
    ),
)

P24 = _t(
    "p24-schema-migration-chain", "long_running",
    "schema.py must upgrade records through every intermediate version. Make tests/test_schema.py pass.",
    {
        "schema.py": (
            "UPGRADES = {\n"
            "    1: lambda r: {**r, 'name': r.pop('title'), 'version': 2},\n"
            "    2: lambda r: {**r, 'tags': [], 'version': 3},\n"
            "    3: lambda r: {**r, 'archived': False, 'version': 4},\n"
            "}\nTARGET = 4\n\n\n"
            "def upgrade(record):\n"
            "    fn = UPGRADES.get(record['version'])\n"
            "    return fn(record) if fn else record\n"
        ),
        "tests/test_schema.py": (
            "import pytest\n\nfrom schema import TARGET, upgrade\n\n\n"
            "def test_single_step():\n"
            "    out = upgrade({'version': 3, 'name': 'a', 'tags': []})\n"
            "    assert out['version'] == TARGET and out['archived'] is False\n\n\n"
            "def test_full_chain():\n"
            "    out = upgrade({'version': 1, 'title': 'a'})\n"
            "    assert out['version'] == TARGET\n"
            "    assert out['name'] == 'a' and out['tags'] == [] and out['archived'] is False\n"
            "    assert 'title' not in out\n\n\n"
            "def test_already_current():\n"
            "    record = {'version': 4, 'name': 'a', 'tags': [], 'archived': False}\n"
            "    assert upgrade(record) == record\n\n\n"
            "def test_unknown_future_version_rejected():\n"
            "    with pytest.raises(ValueError):\n        upgrade({'version': 9})\n"
        ),
    },
    _pytest("tests/test_schema.py"),
    RecordedState(
        branch="feat/schema-chain",
        task_title="upgrade records through every intermediate schema version",
        remaining_work=("loop upgrades until the record reaches TARGET", "raise ValueError for a version above TARGET"),
        decisions=("each step is applied in order; versions are never skipped",),
        verifications=(("schema tests", "failed", "pytest tests/test_schema.py"),),
        changed_paths=("schema.py",),
        rejected=(
            "applying a single upgrade step is the current code and is rejected: a version-1 record comes out at version 2, failing test_full_chain",
            "returning the record unchanged for an unknown version was tried and rejected: test_unknown_future_version_rejected requires a ValueError",
        ),
    ),
)


SUITE: tuple[Task, ...] = (
    P01, P02, P03, P04,      # simple_local
    P05, P06, P07, P08,      # multi_file
    P09, P10, P11, P12,      # repository_wide
    P13, P14, P15, P16,      # state_heavy
    P17, P18, P19, P20,      # failed_hypothesis
    P21, P22, P23, P24,      # long_running
)

assert len(SUITE) == 24, len(SUITE)
assert len({t.task_id for t in SUITE}) == 24, "duplicate task ids"
for _t_ in SUITE:
    assert _t_.recorded.rejected, f"{_t_.task_id} has no recorded trap"

__all__ = ["SUITE"]
