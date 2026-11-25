"""Runtime identity checks without Docker, Ollama, or a native installation."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from benchmarks import pilot_runtime_identity as identity


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", "1")
    package = tmp_path / "site-packages" / "entroly_core"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_bytes(b"from .entroly_core import *\n")
    binary = package / ("entroly_core" + identity.EXTENSION_SUFFIXES[0])
    binary.write_bytes(b"native build one")
    state = SimpleNamespace(
        repo=tmp_path / "repo", engine=SimpleNamespace(path=str(init)),
        url="http://localhost:11434", init=init, binary=binary,
        image="sha256:" + "a" * 64, status="", calls=[],
        body=b'{"version":"0.12.0"}', http_status=200, final_url=None,
        opens=[], reads=[], handlers=(),
        real_run=subprocess.run,
    )

    def run(command, **kwargs):
        state.calls.append((command, kwargs))
        return SimpleNamespace(stdout=state.status if command[0] == "git" else state.image + "\n")

    class Response(io.BytesIO):
        def __init__(self):
            super().__init__(state.body)
            self.status = state.http_status

        def geturl(self):
            return state.final_url or state.url.rstrip("/") + "/api/version"

        def read(self, size=-1):
            state.reads.append(size)
            return super().read(size)

    class Opener:
        def open(self, url, *, timeout):
            state.opens.append((url, timeout))
            return Response()

    def build_opener(*handlers):
        state.handlers = handlers
        return Opener()

    monkeypatch.setattr(identity.subprocess, "run", run)
    monkeypatch.setattr(identity.urllib.request, "build_opener", build_opener)
    monkeypatch.setattr(identity.metadata, "version", lambda name: {"tiktoken": "0.12.0", "regex": "2026.1.1"}[name])
    monkeypatch.setattr(identity.platform, "python_version", lambda: "3.12.7")
    monkeypatch.setattr(identity.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(identity.platform, "system", lambda: "Linux")
    monkeypatch.setattr(identity.platform, "machine", lambda: "x86_64")
    state.collect = lambda: identity.runtime_identity(state.repo, state.engine, state.url)
    return state


def test_stable_complete_identity_has_only_relative_native_paths(runtime):
    first = runtime.collect()
    assert runtime.collect() == first
    assert json.loads(json.dumps(first)) == first
    assert first == {
        "schema": "entroly.pilot-runtime-identity.v1",
        "oracle_image": runtime.image,
        "ollama_version": "0.12.0",
        "native": {"files_sha256": {
            "__init__.py": hashlib.sha256(runtime.init.read_bytes()).hexdigest(),
            runtime.binary.name: hashlib.sha256(runtime.binary.read_bytes()).hexdigest(),
        }},
        "python": {"version": "3.12.7", "implementation": "CPython"},
        "platform": {"system": "Linux", "machine": "x86_64"},
        "distributions": {"tiktoken": "0.12.0", "regex": "2026.1.1"},
    }
    assert str(runtime.init.parent) not in json.dumps(first)
    hashes = first["native"]["files_sha256"]
    assert list(hashes) == sorted(hashes)
    git, docker = runtime.calls[:2]
    assert git == (["git", "-C", str(runtime.repo), "status", "--porcelain",
                    "--untracked-files=normal", "--", "benchmarks", "entroly"],
                   {"check": True, "capture_output": True, "text": True, "timeout": 10})
    assert docker == (["docker", "image", "inspect", "entroly-dogfood-real-oracle:py312",
                       "--format", "{{.Id}}"],
                      {"check": True, "capture_output": True, "text": True, "timeout": 10})
    assert runtime.opens == [(runtime.url + "/api/version", 10)] * 2
    assert runtime.reads == [identity._MAX_VERSION_BYTES + 1] * 2
    assert runtime.handlers[0].proxies == {}
    assert isinstance(runtime.handlers[1], urllib.request.HTTPRedirectHandler)


@pytest.mark.parametrize("value", [None, "", "0", "true", " 1"])
def test_sandbox_must_be_explicit(runtime, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ENTROLY_REAL_COMMIT_SANDBOX")
    else:
        monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", value)
    with pytest.raises(ValueError, match="ENTROLY_REAL_COMMIT_SANDBOX=1"):
        runtime.collect()
    assert runtime.calls == runtime.opens == []


@pytest.mark.parametrize("status", [
    " M benchmarks/real_commit_pilot.py\n", "M  entroly/engine.py\n",
    "?? benchmarks/new.py\n", "?? entroly/new/\n", " D benchmarks/helper.py\n",
])
def test_source_dirt_is_rejected_before_services(runtime, status):
    runtime.status = status
    with pytest.raises(ValueError, match="modified or untracked source"):
        runtime.collect()
    assert len(runtime.calls) == 1
    assert runtime.opens == []


def test_unrelated_research_dirt_is_ignored(runtime, monkeypatch):
    runtime.repo.mkdir()
    runtime.real_run(["git", "init", str(runtime.repo)], check=True, capture_output=True)
    research = runtime.repo / "research"
    research.mkdir()
    (research / "private-notes.txt").write_text("unrelated", encoding="utf-8")
    mocked_run = identity.subprocess.run

    def run(args, **kwargs):
        if args[0] == "git":
            return runtime.real_run(args, **kwargs)
        return mocked_run(args, **kwargs)

    monkeypatch.setattr(identity.subprocess, "run", run)
    assert runtime.collect()["oracle_image"] == runtime.image
    source = runtime.repo / "benchmarks"
    source.mkdir()
    (source / "new.py").write_text("pass", encoding="utf-8")
    with pytest.raises(ValueError, match="modified or untracked source"):
        runtime.collect()


@pytest.mark.parametrize("command", ["git", "docker"])
@pytest.mark.parametrize("failure", ["missing", "failed", "timeout"])
def test_process_failures_fail_closed(runtime, monkeypatch, command, failure):
    run = identity.subprocess.run

    def failing_run(args, **kwargs):
        if args[0] == command:
            assert kwargs["check"] is True
            assert kwargs["timeout"] == 10
            if failure == "missing":
                raise FileNotFoundError("private executable path")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(args, 10)
            raise subprocess.CalledProcessError(1, args, stderr="private details")
        return run(args, **kwargs)

    monkeypatch.setattr(identity.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="unable to") as exc:
        runtime.collect()
    assert "private" not in str(exc.value)
    assert runtime.opens == []


@pytest.mark.parametrize("image", ["", "sha256:abc", "sha256:" + "g" * 64,
                                        "a" * 64, "sha256:" + "a" * 65,
                                        "sha256:" + "a" * 64 + "\nsecond"])
def test_invalid_image_ids_fail_closed(runtime, image):
    runtime.image = image
    with pytest.raises(ValueError, match="immutable sha256 ID"):
        runtime.collect()


def test_changed_image_and_native_bytes_change_identity(runtime):
    first = runtime.collect()
    runtime.image = "sha256:" + "b" * 64
    second = runtime.collect()
    assert second != first
    assert second["oracle_image"] != first["oracle_image"]
    runtime.binary.write_bytes(b"native build two")
    third = runtime.collect()
    assert third["native"] != second["native"]
    runtime.init.write_bytes(b"changed package shim")
    assert runtime.collect()["native"] != third["native"]


def test_package_scan_is_scoped_and_sorted(runtime):
    sibling = runtime.init.parent.parent / ("unrelated" + identity.EXTENSION_SUFFIXES[0])
    sibling.write_bytes(b"must not fingerprint site-packages")
    nested = runtime.init.parent / "nested"
    nested.mkdir()
    binary = nested / ("extra" + identity.EXTENSION_SUFFIXES[0])
    binary.write_bytes(b"package helper binary")
    (runtime.init.parent / "unrelated.dll").write_bytes(b"not a Python extension")
    hashes = runtime.collect()["native"]["files_sha256"]
    assert list(hashes) == sorted(["__init__.py", runtime.binary.name, "nested/" + binary.name])


def test_direct_binary_includes_its_package_init(runtime):
    runtime.engine.path = str(runtime.binary)
    assert "__init__.py" in runtime.collect()["native"]["files_sha256"]


def test_standalone_binary_does_not_scan_sibling_extensions(runtime):
    runtime.init.unlink()
    runtime.engine.path = str(runtime.binary)
    sibling = runtime.binary.parent / ("unrelated" + identity.EXTENSION_SUFFIXES[0])
    sibling.write_bytes(b"unrelated extension")
    assert list(runtime.collect()["native"]["files_sha256"]) == [runtime.binary.name]


@pytest.mark.parametrize("missing", ["binary", "init", "path", "unsupported"])
def test_missing_native_artifacts_fail_closed(runtime, missing):
    if missing == "binary":
        runtime.binary.unlink()
    elif missing == "init":
        runtime.init.unlink()
    elif missing == "path":
        runtime.engine.path = None
    else:
        runtime.engine.path = str(runtime.init.parent / "shim.py")
        (runtime.init.parent / "shim.py").write_text("pass", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError), match="native"):
        runtime.collect()
    assert runtime.opens == []


def test_native_file_outside_module_package_is_rejected(runtime, monkeypatch, tmp_path):
    outside = tmp_path / "other-package" / runtime.binary.name
    real_resolve = type(runtime.binary).resolve

    def resolve(path, *args, **kwargs):
        if path == runtime.binary:
            return outside
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(type(runtime.binary), "resolve", resolve)
    with pytest.raises(ValueError, match="escapes its module package"):
        runtime.collect()


@pytest.mark.parametrize("url", [
    "http://example.com:11434", "http://localhost.example.com:11434",
    "http://127.0.0.1.example.com:11434", "http://0.0.0.0:11434",
    "http://192.168.1.2:11434", "http://[::]:11434", "https://localhost:11434",
    "ftp://localhost:11434", "http://user:secret@localhost:11434",
    "http://localhost:11434/path", "http://localhost:11434?secret=x",
    "http://localhost:11434#fragment", "http://localhost:11434?",
    "http://localhost:11434#", "http://localhost", "http://localhost:0",
    "http://localhost:65536", "http://localhost:notaport", "http://[::1",
    "http://localhost:11434\\@example.com", "http://local\nhost:11434",
])
def test_nonlocal_or_ambiguous_endpoints_do_not_open(runtime, url):
    runtime.url = url
    with pytest.raises(ValueError, match="local Ollama HTTP endpoint"):
        runtime.collect()
    assert runtime.opens == []


@pytest.mark.parametrize("url", ["http://localhost:11434/", "http://127.0.0.1:11434", "http://[::1]:11434"])
def test_local_endpoints(runtime, url):
    runtime.url = url
    assert runtime.collect()["ollama_version"] == "0.12.0"
    assert runtime.opens == [(url.rstrip("/") + "/api/version", 10)]


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_handler_never_follows_a_location(runtime, code):
    runtime.collect()
    redirect = runtime.handlers[1]
    for destination in ("http://remote.example:11434/api/version", runtime.url + "/other"):
        assert redirect.redirect_request(None, None, code, "redirect", {}, destination) is None


@pytest.mark.parametrize("code", [301, 302, 307, 308, 500])
def test_non_success_responses_fail_closed(runtime, code):
    runtime.http_status = code
    with pytest.raises(ValueError, match="must not redirect"):
        runtime.collect()
    assert runtime.reads == []


def test_changed_response_url_is_rejected(runtime):
    runtime.final_url = "http://remote.example/api/version"
    with pytest.raises(ValueError, match="must not redirect"):
        runtime.collect()


@pytest.mark.parametrize("body", [b"{}", b'[]', b'{"version":null}', b'{"version":3}',
                                 b'{"version":""}', b'{"version":"  "}'])
def test_missing_or_invalid_ollama_version(runtime, body):
    runtime.body = body
    with pytest.raises(ValueError, match="nonempty string"):
        runtime.collect()


@pytest.mark.parametrize("body", [b"not JSON", b"\xff"])
def test_invalid_ollama_json(runtime, body):
    runtime.body = body
    with pytest.raises(RuntimeError, match="local Ollama version"):
        runtime.collect()


def test_ollama_response_is_bounded(runtime):
    runtime.body = b" " * (identity._MAX_VERSION_BYTES + 1)
    with pytest.raises(ValueError, match="size limit"):
        runtime.collect()
    assert runtime.reads == [identity._MAX_VERSION_BYTES + 1]


@pytest.mark.parametrize("error", [TimeoutError(), urllib.error.URLError("private endpoint"),
                                    urllib.error.HTTPError("private endpoint", 302, "redirect", {}, None)],
                         ids=["timeout", "unreachable", "redirect"])
def test_ollama_network_errors_fail_closed(runtime, monkeypatch, error):
    def build_opener(*handlers):
        def open_url(*args, **kwargs):
            raise error
        return SimpleNamespace(open=open_url)

    monkeypatch.setattr(identity.urllib.request, "build_opener", build_opener)
    with pytest.raises(RuntimeError, match="local Ollama version") as exc:
        runtime.collect()
    assert "private" not in str(exc.value)


@pytest.mark.parametrize("name", ["tiktoken", "regex"])
def test_missing_optional_distribution_is_frozen_as_absent(runtime, monkeypatch, name):
    def version(distribution):
        if distribution == name:
            raise identity.metadata.PackageNotFoundError(distribution)
        return "1.0"

    monkeypatch.setattr(identity.metadata, "version", version)
    absent = runtime.collect()
    assert absent["distributions"][name] is None
    monkeypatch.setattr(identity.metadata, "version", lambda distribution: "1.0")
    assert runtime.collect() != absent
