"""Local-only benchmark HTTP stays local, including redirects and proxy settings."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from benchmarks.agentic_tasks_run import call_model
from benchmarks.real_commit_pilot import _local_model_digest


@pytest.fixture
def endpoint(monkeypatch):
    state = SimpleNamespace(redirect=False, requests=[], bodies=[])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle_request(self):
            state.requests.append((self.command, self.path))
            if self.command == "POST":
                state.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if state.redirect and self.path != "/unexpected-target":
                self.send_response(302)
                self.send_header("Location", state.url + "/unexpected-target")
                self.end_headers()
                return
            body = json.dumps({
                "models": [{"name": "local-model", "digest": "fixed-digest"}],
                "response": "answer", "prompt_eval_count": 12, "eval_count": 4,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = handle_request
        do_POST = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://unreachable.invalid:1")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_model_metadata_ignores_proxies(endpoint):
    assert _local_model_digest(endpoint.url, "local-model") == "fixed-digest"
    assert endpoint.requests == [("GET", "/api/tags")]


def test_local_generation_preserves_parameters_and_observed_usage(endpoint):
    result = call_model(base_url=endpoint.url, model="local-model", prompt="fixture prompt",
                        seed=17, timeout=2, max_output_tokens=32, context_window=1024,
                        local_only=True)
    assert endpoint.requests == [("POST", "/api/generate")]
    assert endpoint.bodies == [{
        "model": "local-model", "prompt": "fixture prompt", "stream": False,
        "options": {"temperature": 0, "seed": 17, "num_predict": 32, "num_ctx": 1024},
    }]
    assert (result["text"], result["input_tokens"], result["output_tokens"]) == ("answer", 12, 4)


@pytest.mark.parametrize("generation", [False, True])
def test_redirects_are_not_followed_even_to_another_local_path(endpoint, generation):
    endpoint.redirect = True
    if generation:
        with pytest.raises(RuntimeError, match="model call failed"):
            call_model(base_url=endpoint.url, model="local-model", prompt="fixture prompt",
                       seed=17, timeout=2, local_only=True)
    else:
        with pytest.raises(urllib.error.HTTPError):
            _local_model_digest(endpoint.url, "local-model")
    assert len(endpoint.requests) == 1
    assert all(path != "/unexpected-target" for _, path in endpoint.requests)


def test_local_generation_rejects_external_endpoint_before_network():
    with pytest.raises(ValueError, match="local Ollama"):
        call_model(base_url="https://example.invalid:443", model="local-model", prompt="private",
                   seed=17, timeout=2, local_only=True)
