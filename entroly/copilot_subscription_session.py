"""Managed lifecycle for the dedicated Copilot subscription proxy process.

The transport itself remains ``entroly.container_proxy``. This module owns only
process lifetime so a random dedicated port does not leave an orphan proxy after
Copilot CLI exits. Startup reuses the existing validation/health helpers from
``copilot_subscription`` and shutdown is idempotent and bounded.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

from .copilot_subscription import (
    CopilotSubscriptionError,
    CopilotSubscriptionPlan,
    _healthy_proxy,
    _port_is_occupied,
    _runtime_dir,
    _stop_process,
    _tail_log,
    _validate_port,
)


@dataclass(slots=True)
class ManagedCopilotSubscriptionProxy:
    """One dedicated hardened proxy tied to one wrapped Copilot CLI session."""

    process: subprocess.Popen[bytes]
    log_path: Path
    port: int
    _closed: bool = False

    @property
    def running(self) -> bool:
        return not self._closed and self.process.poll() is None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _stop_process(self.process)

    def __enter__(self) -> "ManagedCopilotSubscriptionProxy":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def start_managed_subscription_proxy(
    plan: CopilotSubscriptionPlan,
    *,
    environ: MutableMapping[str, str] | None = None,
    timeout_s: float = 30.0,
) -> ManagedCopilotSubscriptionProxy:
    """Start the hardened proxy and return an explicit lifecycle handle.

    The startup envelope intentionally exceeds the independent bounded auth
    operations (``gh auth token`` plus GitHub's Copilot user/entitlement
    preflight). Network calls keep their tighter per-operation timeouts; only
    the parent health deadline is wider so a slow-but-valid auth path is not
    mistaken for a dead child.
    """
    env = dict(os.environ if environ is None else environ)
    port = _validate_port(plan.proxy_port)
    if _port_is_occupied(port):
        raise CopilotSubscriptionError(
            f"subscription mode requires a dedicated proxy port; {port} is already in use"
        )

    env["ENTROLY_PROXY_HOST"] = "127.0.0.1"
    env["ENTROLY_PROXY_PORT"] = str(port)
    env["ENTROLY_PROXY_DASHBOARD"] = "1"
    env["ENTROLY_OPENAI_BASE"] = plan.upstream_origin
    env["ENTROLY_COPILOT_SUBSCRIPTION"] = "1"

    runtime = _runtime_dir(env)
    log_path = runtime / f"copilot-subscription-proxy-{port}.log"
    command = [sys.executable, "-m", "entroly.container_proxy"]
    try:
        handle = log_path.open("ab")
    except OSError as exc:
        raise CopilotSubscriptionError(
            "unable to create the local Copilot subscription proxy log"
        ) from exc

    try:
        handle.write(
            (
                "\n--- Entroly managed Copilot subscription proxy start "
                f"port={port} ---\n"
            ).encode("utf-8")
        )
        handle.flush()
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise CopilotSubscriptionError(
                "unable to start the hardened Entroly proxy process"
            ) from exc
    finally:
        handle.close()

    deadline = time.monotonic() + max(1.0, float(timeout_s))
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = _tail_log(log_path)
            raise CopilotSubscriptionError(
                "Copilot subscription proxy exited during startup"
                + (f": {detail}" if detail else "")
            )
        if _healthy_proxy(health_url):
            return ManagedCopilotSubscriptionProxy(
                process=process,
                log_path=log_path,
                port=port,
            )
        time.sleep(0.1)

    _stop_process(process)
    detail = _tail_log(log_path)
    raise CopilotSubscriptionError(
        "timed out waiting for the Copilot subscription proxy"
        + (f": {detail}" if detail else "")
    )


__all__ = [
    "ManagedCopilotSubscriptionProxy",
    "start_managed_subscription_proxy",
]
