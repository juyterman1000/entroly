"""The browser extension's two hand-offs to the playground must arrive.

"⚡ Compress Selection with Entroly" saved the selection to
``chrome.storage.local`` and opened the playground. The playground is a web
page on github.io and cannot read extension storage, so the selection never
reached it, and the text stayed in storage indefinitely. The GitHub page
button opened the playground with ``?repo=owner/name``, which the playground
never read. Both landed on the default sample.

The selection now travels in the URL fragment, which the browser does not
send to any server, and the playground reads both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

_BACKGROUND = r"""
const calls = { created: [], stored: [], removed: [] };
const listeners = {};
const on = (name) => ({ addListener: (fn) => { listeners[name] = fn; } });
globalThis.chrome = {
  runtime: { onInstalled: on('installed'), onMessage: on('message') },
  contextMenus: { create() {}, onClicked: on('clicked') },
  storage: { local: {
    set: (value, done) => { calls.stored.push(value); if (done) done(); },
    remove: (key, done) => { calls.removed.push(key); if (done) done(); },
  } },
  tabs: { create: (opts) => calls.created.push(opts.url) },
};
require(process.argv[1]);
listeners.installed && listeners.installed({ reason: 'update' });
listeners.clicked({ menuItemId: 'entroly-compress-selection', selectionText: process.argv[2] }, {});
listeners.message({ type: 'OPEN_PLAYGROUND_WITH_REPO', repo: 'pallets/click' }, {}, () => {});
process.stdout.write(JSON.stringify(calls));
"""


def _background(selection: str) -> dict:
    completed = subprocess.run(
        ["node", "-e", _BACKGROUND, str(ROOT / "extensions" / "chrome" / "background.js"), selection],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_the_selection_travels_in_the_fragment_and_is_not_stored() -> None:
    selection = "def total(a, b):\n    return a + b  # 100% & more"

    calls = _background(selection)

    assert calls["stored"] == [], "selected page text must not be persisted"
    assert "pendingCompressionText" in calls["removed"], "an old copy is cleared"
    url = calls["created"][0]
    assert url.startswith("https://juyterman1000.github.io/entroly/playground/#")
    assert "?" not in url.split("#")[0], "nothing in the query string, which servers log"


_PLAYGROUND = r"""
const vm = require('vm');
const fs = require('fs');
const elements = {};
function element(id) {
  const store = { id, value: '', textContent: '', innerHTML: '', className: '', listeners: {} };
  const proxy = new Proxy(function () {}, {
    get(_t, key) {
      if (key in store) return store[key];
      if (key === 'addEventListener') return (type, fn) => { (store.listeners[type] ||= []).push(fn); };
      if (key === 'click') return () => { for (const fn of store.listeners.click || []) fn({ preventDefault() {} }); };
      if (key === 'classList') return { add() {}, remove() {}, toggle() {}, contains() { return false; } };
      if (key === 'style' || key === 'dataset') return {};
      if (key === 'querySelectorAll') return () => [];
      if (key === 'children') return [];
      if (key === Symbol.toPrimitive) return () => '';
      return proxy;
    },
    set(_t, key, value) { store[key] = value; return true; },
    apply() { return proxy; },
  });
  return proxy;
}
const fetched = [];
let ready = [];
const url = new URL(process.argv[2]);
const replaced = [];
Object.assign(globalThis, {
  window: globalThis,
  location: { search: url.search, hash: url.hash, pathname: url.pathname, href: url.href },
  history: { replaceState: (_s, _t, next) => replaced.push(next) },
  addEventListener() {},
  setTimeout: (fn) => 0, setInterval: () => 0, requestAnimationFrame: () => 0,
  fetch: (target) => { fetched.push(String(target)); return new Promise(() => {}); },
  document: {
    getElementById: (id) => elements[id] || (elements[id] = element(id)),
    querySelector: () => element('q'), querySelectorAll: () => [],
    createElement: () => element('c'),
    addEventListener: (type, fn) => { if (type === 'DOMContentLoaded') ready.push(fn); },
  },
});
Object.defineProperty(globalThis, 'navigator', {
  value: { clipboard: { writeText: async () => {} } }, configurable: true,
});
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));
for (const fn of ready) fn();
setImmediate(() => {
  const corpus = vm.runInThisContext('currentCorpus');
  process.stdout.write(JSON.stringify({
    repoInput: String(elements.githubRepoInput ? elements.githubRepoInput.value : ''),
    fetched, replaced, corpus,
  }));
});
"""


def _playground(url: str) -> dict:
    completed = subprocess.run(
        ["node", "-e", _PLAYGROUND, str(ROOT / "playground" / "engine.js"), url],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_the_playground_loads_the_repository_it_was_opened_for() -> None:
    shown = _playground("https://juyterman1000.github.io/entroly/playground/?repo=pallets%2Fclick")

    assert shown["repoInput"] == "pallets/click"
    assert "https://api.github.com/repos/pallets/click/contents" in shown["fetched"]


def test_the_playground_ignores_a_repo_parameter_that_is_not_owner_and_name() -> None:
    shown = _playground("https://juyterman1000.github.io/entroly/playground/?repo=..%2F..%2Fusers")

    assert shown["fetched"] == []


def test_the_playground_compresses_the_selection_it_was_handed() -> None:
    selection = "def total(a, b):\n    return a + b  # 100% & more"
    calls = _background(selection)

    shown = _playground(calls["created"][0])

    assert [doc["content"] for doc in shown["corpus"]] == [selection]
    assert shown["replaced"] and "#" not in shown["replaced"][-1], (
        "the selection is dropped from the address bar and history once read"
    )
