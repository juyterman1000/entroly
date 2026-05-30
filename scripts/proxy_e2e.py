#!/usr/bin/env python3
"""
End-to-end test for the entroly-rs single-binary proxy (Phase 2).

Spins a threaded mock "upstream", launches the real `entroly-rs proxy` binary
pointed at it, and asserts the behaviours that the unit tests can't cover:
  * /health
  * total-budget request compression (two messages share one budget)
  * response-header forwarding (x-request-id survives)
  * SSE streaming passthrough (text/event-stream, chunks stream through)
  * concurrency (parallel requests all succeed)

Usage:  python scripts/proxy_e2e.py [path-to-entroly-rs]
Requires the binary built with:  cargo build --release --bin entroly-rs --features proxy
Exit code 0 = pass.
"""
import concurrent.futures
import gzip
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BIN = sys.argv[1] if len(sys.argv) > 1 else "entroly-core/target/release/entroly-rs"
MOCK, PROXY = 8773, 8774
recv = {}


class Mock(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        recv[self.path] = len(body)
        compact = body.replace(b" ", b"")
        if b'"stream":true' in compact:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("x-request-id", "req-123")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: chunk{i}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.03)
        else:
            # GZIP the response, like real Anthropic. ureq decompresses it; the
            # proxy must re-serve clean decoded JSON (no content-encoding, no
            # stale compressed content-length) — this catches the gzip framing
            # bug a plain-text mock would miss.
            raw = json.dumps({"ok": True, "received_bytes": len(body)}).encode()
            gz = gzip.compress(raw)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("x-request-id", "req-123")
            self.send_header("Content-Length", str(len(gz)))
            self.end_headers()
            self.wfile.write(gz)

    def log_message(self, *a):
        pass


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", MOCK), Mock)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    proxy = subprocess.Popen(
        [BIN, "proxy", "--port", str(PROXY), "--upstream", f"http://127.0.0.1:{MOCK}", "--budget", "80"],
        stderr=subprocess.PIPE,
    )
    time.sleep(1.5)
    try:
        # health
        h = urllib.request.urlopen(f"http://127.0.0.1:{PROXY}/health", timeout=5).read().decode()
        assert "ok" in h, h
        print("health OK")

        big = "\n".join(f"Line {i}: real content about module {i} edge cases." for i in range(300))

        def post(path, obj):
            data = json.dumps(obj).encode()
            r = urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{PROXY}{path}", data=data,
                    headers={"Content-Type": "application/json", "x-api-key": "t"},
                ),
                timeout=10,
            )
            return data, r

        # Anthropic — total budget across two messages + gzip decode + header fwd.
        payload, r = post("/v1/messages", {"messages": [
            {"role": "user", "content": big},
            {"role": "user", "content": big},
        ]})
        assert r.headers.get("x-request-id") == "req-123", "response header must be forwarded"
        assert recv["/v1/messages"] < len(payload), "anthropic must compress (total budget)"
        assert json.loads(r.read().decode()).get("ok") is True, "gzip response must decode cleanly"
        assert r.headers.get("Content-Encoding") is None, "content-encoding must be stripped"
        print(f"anthropic: {len(payload)} -> {recv['/v1/messages']} (gzip decoded + header fwd) OK")

        # OpenAI — /chat/completions (string + text-part content).
        op, _ = post("/v1/chat/completions", {"model": "gpt-4o", "messages": [
            {"role": "system", "content": big},
            {"role": "user", "content": [{"type": "text", "text": big}]},
        ]})
        assert recv["/v1/chat/completions"] < len(op), "openai must compress"
        print(f"openai: {len(op)} -> {recv['/v1/chat/completions']} OK")

        # Gemini — generateContent (contents.parts + systemInstruction).
        gpath = "/v1beta/models/gemini-2.5-pro:generateContent"
        gp, _ = post(gpath, {
            "contents": [{"role": "user", "parts": [{"text": big}]}],
            "systemInstruction": {"parts": [{"text": big}]},
        })
        assert recv[gpath] < len(gp), "gemini must compress"
        print(f"gemini: {len(gp)} -> {recv[gpath]} OK")

        # SSE streaming passthrough
        spayload = json.dumps({"stream": True, "messages": [{"role": "user", "content": big}]}).encode()
        sreq = urllib.request.Request(
            f"http://127.0.0.1:{PROXY}/v1/messages", data=spayload,
            headers={"Content-Type": "application/json"},
        )
        sr = urllib.request.urlopen(sreq, timeout=10)
        assert "event-stream" in sr.headers.get("Content-Type", ""), "SSE content-type must pass through"
        sbody = sr.read().decode()
        assert "chunk0" in sbody and "chunk2" in sbody, "SSE chunks must stream through"
        print("SSE streaming passthrough OK")

        # concurrency
        def one(_):
            rq = urllib.request.Request(
                f"http://127.0.0.1:{PROXY}/v1/messages", data=payload,
                headers={"Content-Type": "application/json"},
            )
            return urllib.request.urlopen(rq, timeout=15).status
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            codes = list(ex.map(one, range(12)))
        assert all(c == 200 for c in codes), f"all parallel requests must succeed: {codes}"
        print("concurrency: 12/12 parallel requests OK")

        print("PASS: total-budget compress + header forward + SSE stream + concurrency")
        return 0
    finally:
        proxy.terminate()
        httpd.shutdown()


if __name__ == "__main__":
    sys.exit(main())
