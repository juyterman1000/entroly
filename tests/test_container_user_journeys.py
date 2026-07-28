from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import entroly.container_entry as container_entry
import entroly.container_health as container_health
import entroly.container_proxy as container_proxy
import entroly.docker_launcher_safe as safe_launcher


ROOT = Path(__file__).resolve().parents[1]


def test_safe_launcher_routes_proxy_serve_without_docker(monkeypatch) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["entroly", "serve", "--proxy", "--port", "9444"])
    monkeypatch.setattr(
        safe_launcher,
        "_run_native_proxy",
        lambda argv: observed.append(list(argv)),
    )
    monkeypatch.setattr(
        safe_launcher._legacy,
        "launch",
        lambda: pytest.fail("proxy mode must not enter the Docker MCP launcher"),
    )

    safe_launcher.launch()

    assert observed == [["serve", "--proxy", "--port", "9444"]]


def test_safe_launcher_delegates_stdio_mcp_serve_to_versioned_docker(monkeypatch) -> None:
    called = False

    def legacy_launch() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(sys, "argv", ["entroly", "serve"])
    monkeypatch.setattr(safe_launcher._legacy, "launch", legacy_launch)

    safe_launcher.launch()

    assert called


def test_native_proxy_cli_overrides_are_explicit_and_loopback_by_default(
    monkeypatch,
) -> None:
    observed: dict[str, str] = {}

    def proxy_main() -> None:
        observed["host"] = safe_launcher.os.environ["ENTROLY_PROXY_HOST"]
        observed["port"] = safe_launcher.os.environ["ENTROLY_PROXY_PORT"]

    monkeypatch.delenv("ENTROLY_PROXY_HOST", raising=False)
    monkeypatch.delenv("ENTROLY_PROXY_PORT", raising=False)
    monkeypatch.setattr(container_proxy, "main", proxy_main)

    safe_launcher._run_native_proxy(["serve", "--proxy", "--port", "9445"])

    assert observed == {"host": "127.0.0.1", "port": "9445"}


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--proxy", "--port"],
        ["serve", "--proxy", "--unknown"],
        ["serve", "--proxy", "positional"],
    ],
)
def test_native_proxy_rejects_ignored_or_incomplete_arguments(argv) -> None:
    with pytest.raises(ValueError, match="requires a value|unsupported"):
        safe_launcher._apply_proxy_cli_overrides(argv)


def test_container_entry_dispatches_proxy_without_ignored_serve_token(monkeypatch) -> None:
    called = False

    def proxy_main() -> None:
        nonlocal called
        called = True

    monkeypatch.delenv("ENTROLY_CONTAINER_MODE", raising=False)
    monkeypatch.delenv("ENTROLY_PROXY_HOST", raising=False)
    monkeypatch.setattr(container_proxy, "main", proxy_main)

    container_entry.main(["serve", "--proxy"])

    assert called
    assert container_entry.os.environ["ENTROLY_CONTAINER_MODE"] == "proxy"
    assert container_entry.os.environ["ENTROLY_PROXY_HOST"] == "127.0.0.1"


def test_container_entry_dispatches_default_stdio_and_restores_argv(monkeypatch) -> None:
    import entroly.server as server

    observed: list[list[str]] = []
    original = ["pytest", "original"]
    monkeypatch.setattr(sys, "argv", original.copy())
    monkeypatch.setattr(server, "main", lambda: observed.append(list(sys.argv)))
    monkeypatch.delenv("ENTROLY_CONTAINER_MODE", raising=False)

    container_entry.main(["serve"])

    assert observed == [["pytest"]]
    assert sys.argv == original
    assert container_entry.os.environ["ENTROLY_CONTAINER_MODE"] == "mcp"


def test_container_entry_preserves_supported_sse_mode(monkeypatch) -> None:
    import entroly.server as server

    observed: list[list[str]] = []
    monkeypatch.setattr(server, "main", lambda: observed.append(list(sys.argv)))

    container_entry.main(["--sse"])

    assert observed and observed[0][-1] == "--sse"


@pytest.mark.parametrize(
    "argv",
    [
        ["--proxy", "--ignored"],
        ["serve", "unexpected"],
        ["--unknown"],
    ],
)
def test_container_entry_rejects_unknown_arguments(argv) -> None:
    with pytest.raises(SystemExit, match="unsupported"):
        container_entry.main(argv)


def test_proxy_health_uses_capability_and_validates_service_identity(monkeypatch) -> None:
    observed = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"status": "ok", "service": "entroly-proxy"}
            ).encode("utf-8")

    def urlopen(request, timeout):
        observed["token"] = request.get_header("X-entroly-access-token")
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setenv("ENTROLY_PROXY_PORT", "9446")
    monkeypatch.setenv("ENTROLY_PROXY_ACCESS_TOKEN", "t" * 48)
    monkeypatch.setattr(container_health.urllib.request, "urlopen", urlopen)

    assert container_health._proxy_healthy()
    assert observed == {
        "token": "t" * 48,
        "url": "http://127.0.0.1:9446/health",
        "timeout": 3,
    }


def test_proxy_health_rejects_wrong_service_and_invalid_port(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"status":"ok","service":"attacker"}'

    monkeypatch.setenv("ENTROLY_PROXY_PORT", "9377")
    monkeypatch.setattr(
        container_health.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    assert not container_health._proxy_healthy()

    monkeypatch.setenv("ENTROLY_PROXY_PORT", "not-a-port")
    assert not container_health._proxy_healthy()


def test_container_health_selects_the_actual_mode(monkeypatch) -> None:
    monkeypatch.setattr(container_health, "_proxy_mode", lambda: True)
    monkeypatch.setattr(container_health, "_proxy_healthy", lambda: True)
    monkeypatch.setattr(
        container_health,
        "_mcp_process_healthy",
        lambda: pytest.fail("proxy mode must not use MCP health"),
    )
    assert container_health.main() == 0

    monkeypatch.setattr(container_health, "_proxy_mode", lambda: False)
    monkeypatch.setattr(container_health, "_mcp_process_healthy", lambda: False)
    assert container_health.main() == 1


def test_container_proxy_validates_app_before_background_threads(monkeypatch) -> None:
    import uvicorn

    events: list[object] = []
    config = SimpleNamespace(host="127.0.0.1", port=9377)
    engine = SimpleNamespace(checkpoint=lambda: None)
    app = object()

    monkeypatch.setattr(container_proxy.ProxyConfig, "from_env", lambda: config)
    monkeypatch.setattr(container_proxy, "EntrolyEngine", lambda: engine)
    monkeypatch.setattr(
        container_proxy,
        "create_proxy_app",
        lambda *_args, **kwargs: events.append(("app", kwargs)) or app,
    )
    monkeypatch.setattr(
        container_proxy,
        "_start_background_services",
        lambda selected: events.append(("background", selected)),
    )
    monkeypatch.setattr(container_proxy.atexit, "register", lambda *_args: None)
    monkeypatch.setattr(container_proxy.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda selected, **kwargs: events.append(("uvicorn", selected, kwargs)),
    )
    monkeypatch.delenv("ENTROLY_PROXY_DASHBOARD", raising=False)

    container_proxy.main()

    assert events[0] == (
        "app",
        {"start_dashboard": False, "start_autotune": False},
    )
    assert events[1] == ("background", engine)
    assert events[2][0] == "uvicorn"
    assert events[2][1] is app


def test_container_proxy_rejects_invalid_dashboard_flag_before_engine(monkeypatch) -> None:
    config = SimpleNamespace(host="127.0.0.1", port=9377)
    monkeypatch.setattr(container_proxy.ProxyConfig, "from_env", lambda: config)
    monkeypatch.setenv("ENTROLY_PROXY_DASHBOARD", "maybe")
    monkeypatch.setattr(
        container_proxy,
        "EntrolyEngine",
        lambda: pytest.fail("invalid config must fail before engine creation"),
    )

    with pytest.raises(ValueError, match="ENTROLY_PROXY_DASHBOARD"):
        container_proxy.main()


def test_base_wheel_declares_every_public_proxy_runtime() -> None:
    root = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    nested = (ROOT / "entroly" / "pyproject.toml").read_text(encoding="utf-8")
    for manifest in (root, nested):
        dependency_block = manifest.split("dependencies = [", 1)[1].split("]", 1)[0]
        assert '"mcp>=1.28.1,<2"' in dependency_block
        assert '"httpx>=0.28.1"' in dependency_block
        assert '"starlette>=1.3.1"' in dependency_block
        assert '"uvicorn>=0.51.0"' in dependency_block
        assert 'entroly = "entroly.docker_launcher_safe:launch"' in manifest


def test_dockerfile_uses_truthful_dispatch_and_mode_aware_health() -> None:
    dockerfile = (ROOT / "Dockerfile.entroly").read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["python", "-m", "entroly.container_entry"]' in dockerfile
    assert 'CMD ["python", "-m", "entroly.container_health"]' in dockerfile
    assert "entroly.server --proxy" not in dockerfile
    assert "httpx.get('http://localhost:9377/health')" not in dockerfile
