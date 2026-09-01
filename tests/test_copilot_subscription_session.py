from __future__ import annotations

from types import SimpleNamespace

import pytest

from entroly.copilot_subscription import CopilotSubscriptionPlan
from entroly.copilot_subscription_session import (
    ManagedCopilotSubscriptionProxy,
    start_managed_subscription_proxy,
)


def test_managed_proxy_close_is_idempotent(monkeypatch, tmp_path) -> None:
    calls: list[object] = []
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(
        "entroly.copilot_subscription_session._stop_process",
        lambda proc: calls.append(proc),
    )
    managed = ManagedCopilotSubscriptionProxy(
        process=process,  # type: ignore[arg-type]
        log_path=tmp_path / "proxy.log",
        port=19477,
    )

    assert managed.running is True
    managed.close()
    managed.close()

    assert calls == [process]
    assert managed.running is False


def test_managed_proxy_start_returns_owned_process(monkeypatch, tmp_path) -> None:
    class FakeProcess:
        def poll(self):
            return None

    process = FakeProcess()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "entroly.copilot_subscription_session._port_is_occupied",
        lambda _port: False,
    )
    monkeypatch.setattr(
        "entroly.copilot_subscription_session._runtime_dir",
        lambda _env: tmp_path,
    )
    monkeypatch.setattr(
        "entroly.copilot_subscription_session._healthy_proxy",
        lambda _url: True,
    )

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return process

    monkeypatch.setattr(
        "entroly.copilot_subscription_session.subprocess.Popen",
        fake_popen,
    )

    plan = CopilotSubscriptionPlan(
        cleaned_argv=("wrap", "copilot", "--port", "19477"),
        upstream_origin="https://api.githubcopilot.com",
        wire_api="completions",
        model="gpt-5",
        proxy_port=19477,
    )
    managed = start_managed_subscription_proxy(plan, environ={})

    assert managed.process is process
    assert managed.port == 19477
    assert captured["command"][-2:] == ["-m", "entroly.container_proxy"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["ENTROLY_PROXY_HOST"] == "127.0.0.1"
    assert env["ENTROLY_PROXY_PORT"] == "19477"
    assert env["ENTROLY_COPILOT_SUBSCRIPTION"] == "1"


def test_launcher_closes_managed_proxy_when_wrapped_cli_exits(monkeypatch) -> None:
    import entroly.docker_launcher_safe as launcher

    class FakeManaged:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    managed = FakeManaged()
    monkeypatch.setattr(
        launcher,
        "_prepare_copilot_subscription",
        lambda _argv: (True, managed),
    )
    monkeypatch.setattr(launcher.sys, "argv", ["entroly", "wrap", "copilot"])

    def exit_from_cli():
        raise SystemExit(7)

    monkeypatch.setattr(launcher._legacy, "launch", exit_from_cli)

    with pytest.raises(SystemExit) as exc:
        launcher.launch()

    assert exc.value.code == 7
    assert managed.closed is True
