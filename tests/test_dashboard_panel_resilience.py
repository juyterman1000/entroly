"""A polled panel must not discard correct data because one poll failed.

Reported from a live dashboard: "sometimes it does not display anything and
most of the time it's displaying correctly", alongside "Context health
unavailable: Failed to fetch".

Each panel refreshes every 3 seconds and, on any error, replaced its contents
with the error text. A single transient fetch rejection therefore wiped a panel
that had been correct a moment earlier, and replaced it with a message naming
neither the endpoint nor a cause a user could act on. The blanking was the
visible symptom; the discarded reading was the actual loss.

Exercised through Node against the shipped script rather than a copy, so the
test cannot drift from what the browser runs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is required")


def _panel_logic() -> str:
    from entroly.dashboard import DASHBOARD_HTML

    blocks = re.findall(r"<script[^>]*>(.*?)</script>", DASHBOARD_HTML, re.S)
    script = "\n;\n".join(blocks)
    start = script.index("const PANEL_FAILURES_BEFORE_BLANKING")
    end = script.index("async function refreshContextHealth")
    return script[start:end]


HARNESS = """
global.document = { createElement: () => ({className:"", textContent:"", remove(){}}) };
function escHtml(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
__PANEL__
function el(){return {innerHTML:"", _kids:[],
  querySelector(sel){return this._kids.find(k=>("."+k.className).includes(sel.replace(".","")))||null;},
  prepend(n){this._kids.push(n);}};}
const out = [];
let panel = el(); panel.innerHTML = "<div>GOOD DATA</div>";
panelSucceeded("p");
for (let i=1;i<=3;i++){
  const blanked = panelFailed("p", panel, "Context health", new Error("Failed to fetch"));
  out.push({attempt:i, blanked:blanked, kept:panel.innerHTML.includes("GOOD DATA"), notes:panel._kids.length});
}
let fresh = el();
out.push({fresh:true, blanked:panelFailed("q", fresh, "Idx", new Error("Failed to fetch")),
          names:fresh.innerHTML.includes("Failed to fetch")});
let rec = el(); rec.innerHTML="<div>G</div>";
panelSucceeded("r"); panelFailed("r", rec, "X", new Error("b"));
panelClearStale(rec); panelSucceeded("r");
out.push({recovered:true, blanked:panelFailed("r", rec, "X", new Error("b"))});
console.log(JSON.stringify(out));
"""


def _run() -> list[dict]:
    import json

    script = HARNESS.replace("__PANEL__", _panel_logic())
    path = Path(tempfile.mkdtemp()) / "panel.js"
    path.write_text(script, encoding="utf-8")
    done = subprocess.run(
        [NODE, str(path)], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(180)
def test_a_single_failed_poll_does_not_blank_a_good_panel():
    first = _run()[0]
    assert first["blanked"] is False
    assert first["kept"] is True, (
        "one transient fetch rejection destroyed a reading that was correct; "
        "this is the blank dashboard users reported"
    )


@pytest.mark.timeout(180)
def test_stale_data_is_labelled_not_shown_silently():
    first = _run()[0]
    assert first["notes"] == 1, (
        "kept data must carry a staleness note -- quietly showing an old "
        "reading as current is the opposite failure"
    )


@pytest.mark.timeout(180)
def test_repeated_staleness_does_not_stack_notes():
    assert _run()[1]["notes"] == 1


@pytest.mark.timeout(180)
def test_persistent_failure_eventually_replaces_the_reading():
    third = _run()[2]
    assert third["blanked"] is True, (
        "data that has not refreshed for three consecutive attempts should no "
        "longer be presented as the state of the system"
    )


@pytest.mark.timeout(180)
def test_a_panel_that_never_loaded_reports_immediately():
    fresh = _run()[3]
    assert fresh["blanked"] is True
    assert fresh["names"] is True, "the message must name the cause"


@pytest.mark.timeout(180)
def test_recovery_resets_the_failure_count():
    assert _run()[4]["blanked"] is False, (
        "a panel that recovered must get the full allowance again, or a "
        "long-lived dashboard blanks on the first hiccup after an old one"
    )
