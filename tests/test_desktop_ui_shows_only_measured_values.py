"""The desktop control plane must not print figures it never measured.

`ui/app.js` is embedded in the Entroly Desktop binary. It started with
"1,428,500" tokens pruned, "$14.28" banked and "0.22ms p95" latency, and it
kept those numbers whether or not the dashboard answered: the fields it read
(`tokens_reduced`, `banked_usd`, `latency_ms` at the top level) are not in
`/api/stats`, and the offline branch set the status to "DAEMON :9377 ACTIVE"
like the online one. Its "live" compression preview reported
`rawTokens * 0.22` as the compressed size, so any pasted text showed about
78% saved, under a "0 semantic distortion" banner.

These tests run the page's own script in Node with a stub DOM.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "ui" / "app.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

_HARNESS = r"""
const els = {};
function element(id) {
  const listeners = {};
  const store = { id, textContent: '', innerHTML: '', value: '', listeners };
  const stub = () => proxy;
  const proxy = new Proxy(stub, {
    get(_t, key) {
      if (key in store) return store[key];
      if (key === 'addEventListener') return (type, fn) => { (listeners[type] ||= []).push(fn); };
      if (key === 'classList') return { add() {}, remove() {}, toggle() {}, contains() { return false; } };
      if (key === 'style' || key === 'dataset') return {};
      if (key === 'querySelectorAll') return () => [];
      if (key === Symbol.toPrimitive) return () => '';
      return proxy;
    },
    set(_t, key, value) { store[key] = value; return true; },
    apply() { return proxy; },
  });
  return proxy;
}
let ready;
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.navigator = { clipboard: { writeText: async () => {} } };
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.document = {
  getElementById: (id) => els[id] || (els[id] = element(id)),
  querySelector: () => element('q'),
  querySelectorAll: () => [],
  createElement: () => element('c'),
  addEventListener: (type, fn) => { if (type === 'DOMContentLoaded') ready = fn; },
  body: element('body'),
  documentElement: element('html'),
};
globalThis.setInterval = () => 0;
globalThis.setTimeout = (fn) => { fn(); return 0; };
globalThis.requestAnimationFrame = () => 0;
const MODE = process.argv[1];
globalThis.fetch = MODE === 'offline'
  ? () => Promise.reject(new Error('connection refused'))
  : () => Promise.resolve({ ok: true, json: async () => JSON.parse(process.argv[2]) });
require(process.argv[3]);
ready();
setImmediate(() => {
  const out = {};
  for (const id of ['statusText', 'latencyText', 'topTokensPruned', 'topBankedUsd']) {
    out[id] = String(els[id].textContent);
  }
  if (process.argv[4]) {
    els.inputEditor.value = process.argv[4];
    for (const fn of els.btnCompress.listeners.click || []) fn({ preventDefault() {} });
    out.savingsRatio = String(els.savingsRatio.textContent);
    out.compressedCount = String(els.compressedCount.textContent);
    out.outputPreview = String(els.outputPreview.innerHTML);
    out.toast = String(els.toastNotification.textContent);
  }
  process.stdout.write(JSON.stringify(out));
});
"""


def _run(mode: str, payload: dict | None = None, paste: str = "") -> dict:
    completed = subprocess.run(
        ["node", "-e", _HARNESS, mode, json.dumps(payload or {}), str(APP_JS), paste],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_offline_the_header_says_offline_and_shows_no_figures() -> None:
    shown = _run("offline")

    assert "OFFLINE" in shown["statusText"]
    assert "ACTIVE" not in shown["statusText"]
    assert shown["latencyText"] == "—"
    assert shown["topTokensPruned"] == "—"
    assert shown["topBankedUsd"] == "—"


def test_online_the_header_shows_what_the_dashboard_reported() -> None:
    shown = _run(
        "online",
        {"banked_value": {"tokens_reduced": 5321, "modeled_future_value_usd": 0.016}},
    )

    assert "ACTIVE" in shown["statusText"]
    assert shown["topTokensPruned"] == "5,321"
    assert "1,428,500" not in json.dumps(shown)
    assert "14.28" not in shown["topBankedUsd"]
    assert shown["latencyText"] == "—", "the dashboard reports no latency"


def test_the_preview_reports_what_its_rules_removed_not_a_fixed_ratio() -> None:
    plain = "x = 1\ny = 2\nprint(x + y)\n"

    shown = _run("offline", paste=plain)

    assert shown["savingsRatio"].startswith("0.0%"), shown["savingsRatio"]
    assert "0 semantic distortion" not in shown["outputPreview"]
    assert "78.0%" not in shown["toast"] and "0.22ms" not in shown["toast"], shown["toast"]
    assert "0.0%" in shown["toast"]
