"""Quick smoke test for the controls page and control API."""
import json
import time
import urllib.request

from entroly.dashboard import start_dashboard
from entroly.server import EntrolyEngine

e = EntrolyEngine()
s = start_dashboard(e, port=19378, daemon=True)
time.sleep(1)

tests = 0
passed = 0

def check(name, url, expect_key=None, method="GET", body=None):
    global tests, passed
    tests += 1
    try:
        if method == "POST":
            data = json.dumps(body).encode() if body else b"{}"
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        else:
            req = url
        r = urllib.request.urlopen(req, timeout=3)
        d = json.loads(r.read()) if expect_key else None
        if expect_key and expect_key not in (d if isinstance(d, dict) else {}):
            print(f"  FAIL {name}: missing key '{expect_key}' in {d}")
            return
        passed += 1
        val = d.get(expect_key, "") if d else r.status
        print(f"  OK   {name} => {val}")
    except Exception as ex:
        print(f"  FAIL {name}: {ex}")

# GET routes
check("/controls page", "http://localhost:19378/controls")
check("control/status", "http://localhost:19378/api/control/status", "status")
check("control/repos", "http://localhost:19378/api/control/repos", "repos")
check("control/learning", "http://localhost:19378/api/control/learning", "local_enabled")
check("control/federation", "http://localhost:19378/api/control/federation", "enabled")
check("control/logs", "http://localhost:19378/api/control/logs", "lines")

# POST routes
check("POST optimization/pause", "http://localhost:19378/api/control/optimization/pause", "ok", "POST")
check("POST optimization/enable", "http://localhost:19378/api/control/optimization/enable", "ok", "POST")
check("POST bypass", "http://localhost:19378/api/control/bypass", "ok", "POST", {"enabled": True})
check("POST quality", "http://localhost:19378/api/control/quality", "ok", "POST", {"mode": "max"})
check("POST learning/enable", "http://localhost:19378/api/control/learning/enable", "ok", "POST", {"enabled": True})
check("POST learning/reset", "http://localhost:19378/api/control/learning/reset", "ok", "POST")
check("POST repos/reindex", "http://localhost:19378/api/control/repos/reindex", "ok", "POST")

s.shutdown()
print(f"\n  {passed}/{tests} passed")
