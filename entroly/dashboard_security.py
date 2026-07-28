"""Security boundary for Entroly's localhost dashboard and control API.

The dashboard displays repository paths, prompts, model output, security
findings, selection reasons, telemetry, and exception messages. Loopback is not
a trust boundary: a malicious repository can persist HTML into those values and
a browser can send requests to localhost from an unrelated website.

This module is installed after :mod:`entroly.dashboard` loads and preserves the
public ``start_dashboard`` API while enforcing:

* display-safe, bounded JSON for routes rendered through ``innerHTML``;
* a per-process capability token for every state-changing control request;
* loopback client, exact Host, Origin, Fetch-Metadata, method, and body checks;
* socket timeouts, a bounded accept queue, and a capped request-worker pool;
* fail-closed startup when the embedded controls page cannot be hardened.
"""

from __future__ import annotations

import hmac
import html
import ipaddress
import json
import logging
import math
import os
import secrets
import sys
import threading
import time
from http.server import HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import urlsplit

from . import controls_html as _controls
from . import dashboard as _dashboard

logger = logging.getLogger("entroly.dashboard.security")

_DEFAULT_MAX_WORKERS = 24
_DEFAULT_SOCKET_TIMEOUT_S = 5.0
_MAX_REQUEST_TARGET_CHARS = 8192
_MAX_TOKEN_CHARS = 512
_MAX_DISPLAY_DEPTH = 12
_MAX_DISPLAY_ITEMS = 1000
_MAX_DISPLAY_STRING_CHARS = 8192
_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DISPLAY_SAFE_PATHS = frozenset(
    {
        "/api/metrics",
        "/api/trends",
        "/api/confidence",
        "/api/control/status",
        "/api/control/repos",
        "/api/control/learning",
        "/api/control/privacy",
        "/api/control/federation",
        "/api/control/context/last",
        "/api/control/logs",
    }
)
_INSTALLED = False
_HARDENING_ERRORS: tuple[str, ...] = ()


def _validated_control_token(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    if not 32 <= len(raw) <= _MAX_TOKEN_CHARS:
        return None
    if any(ord(character) < 33 or ord(character) == 127 for character in raw):
        return None
    return raw


def _load_control_token() -> str:
    configured = os.environ.get("ENTROLY_CONTROL_TOKEN")
    if configured is not None:
        validated = _validated_control_token(configured)
        if validated is not None:
            return validated
        logger.warning(
            "Ignoring invalid ENTROLY_CONTROL_TOKEN; expected 32-512 visible characters"
        )
    return secrets.token_urlsafe(32)


CONTROL_TOKEN = _load_control_token()


def control_token() -> str:
    """Return the active token for trusted in-process clients and tests."""
    return CONTROL_TOKEN


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if not minimum <= value <= maximum:
        logger.warning("Out-of-range %s=%r; using %d", name, raw, default)
        return default
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default
    if not math.isfinite(value) or not minimum <= value <= maximum:
        logger.warning("Out-of-range %s=%r; using %.1f", name, raw, default)
        return default
    return value


def _bounded_display_string(value: str) -> str:
    if len(value) <= _MAX_DISPLAY_STRING_CHARS:
        return html.escape(value, quote=True)
    return html.escape(value[:_MAX_DISPLAY_STRING_CHARS], quote=True) + "…[truncated]"


def _display_safe(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    """Copy JSON-like data into bounded HTML-safe display values.

    Numbers and booleans retain their JSON types. Strings become HTML text, not
    markup. Cycles, pathological nesting, and giant collections are represented
    explicitly instead of exhausting the dashboard process.
    """
    if depth > _MAX_DISPLAY_DEPTH:
        return "[depth-limit]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else 0.0
    if isinstance(value, str):
        return _bounded_display_string(value)

    tracked = isinstance(value, (dict, list, tuple, set))
    active = seen if seen is not None else set()
    identity = id(value)
    if tracked:
        if identity in active:
            return "[cycle]"
        active.add(identity)
    try:
        if isinstance(value, dict):
            output: dict[str, Any] = {}
            for index, (key, child) in enumerate(value.items()):
                if index >= _MAX_DISPLAY_ITEMS:
                    output["_truncated"] = True
                    break
                output[str(key)] = _display_safe(
                    child,
                    depth=depth + 1,
                    seen=active,
                )
            return output
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            output = [
                _display_safe(child, depth=depth + 1, seen=active)
                for child in items[:_MAX_DISPLAY_ITEMS]
            ]
            if len(items) > _MAX_DISPLAY_ITEMS:
                output.append("[items-truncated]")
            return output
        return _bounded_display_string(str(value))
    finally:
        if tracked:
            active.discard(identity)


def _inject_once(source: str, marker: str, payload: str, *, label: str) -> tuple[str, str | None]:
    count = source.count(marker)
    if count != 1:
        return source, f"{label}: expected one {marker!r}, found {count}"
    return source.replace(marker, payload + marker), None


_SAFE_CONTROLS_SCRIPT = """
<script>
(() => {
  'use strict';
  const CONTROL_TOKEN = __ENTROLY_CONTROL_TOKEN__;
  const escHtml = value => String(value == null ? '' : value).replace(
    /[&<>"']/g,
    character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character])
  );
  const decodeHtml = value => {
    const node = document.createElement('textarea');
    node.innerHTML = String(value == null ? '' : value);
    return node.value;
  };
  const jsArg = value => escHtml(JSON.stringify(String(value == null ? '' : value)));

  window.ctrlPost = async function(url, body = {}, message = 'Done') {
    try {
      controlError('');
      const response = await fetch(url, {
        method: 'POST',
        credentials: 'same-origin',
        headers: {
          'Content-Type': 'application/json',
          'X-Entroly-Control-Token': CONTROL_TOKEN,
        },
        body: JSON.stringify(body),
      });
      let payload = {};
      try {
        payload = await response.json();
      } catch (_error) {
        payload = {ok: false, error: 'Invalid server response'};
      }
      if (response.ok && payload.ok) {
        toast(message);
        return payload;
      }
      const detail = payload.error || ('HTTP ' + response.status + ' from control API');
      controlError(detail);
      toast(detail, false);
      return payload;
    } catch (error) {
      const detail = 'Connection error: ' + (error.message || String(error));
      controlError(detail);
      toast(detail, false);
      return {ok: false, error: detail};
    } finally {
      refresh();
    }
  };

  window.renderRepos = function(repos) {
    if (!repos || !repos.length) {
      return '<div style="color:var(--dim);font-size:13px;">No repos configured</div>';
    }
    return repos.map(repo => {
      const path = decodeHtml(repo.path || '');
      const sync = repo.last_sync
        ? ' &middot; synced ' + new Date(repo.last_sync * 1000).toLocaleTimeString()
        : '';
      return `<div class="repo-item"><span class="repo-icon">${repo.watching ? '&#128994;' : '&#128308;'}</span>
<div class="repo-info"><div class="repo-path">${escHtml(path)}</div>
<div class="repo-meta">${Number(repo.indexed_files || 0)} files &middot; ${Number(repo.total_tokens || 0).toLocaleString()} tokens${sync}</div></div>
<button class="btn" onclick="ctrlPost('/api/control/repos/reindex',{path:${jsArg(path)}})" style="flex-shrink:0;">Re-index</button></div>`;
    }).join('');
  };

  window.renderContext = function(context) {
    if (!context || (!context.included && !context.excluded)) {
      return '<div style="color:var(--dim);font-size:13px;">No context data yet</div>';
    }
    const included = context.included || [];
    const excluded = context.excluded || [];
    let output = '<div style="font-size:12px;color:var(--dim);margin-bottom:8px;">' +
      included.length + ' included &middot; ' + excluded.length + ' excluded</div>';
    included.slice(0, 8).forEach(fragment => {
      const source = decodeHtml(fragment.source || fragment.id || '').split(/[\\/]/).pop();
      const score = Number(((fragment.scores || {}).composite) || 0);
      output += `<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid var(--border);">
<span style="color:var(--emerald);">&#10003; ${escHtml(source)}</span><span style="color:var(--dim);font-family:'JetBrains Mono',monospace;">${(score * 100).toFixed(1)}%</span></div>`;
    });
    return output;
  };

  window.refreshLogs = async function() {
    const viewer = document.getElementById('logViewer');
    try {
      const response = await fetch('/api/control/logs', {credentials: 'same-origin'});
      const payload = await response.json();
      viewer.replaceChildren();
      const lines = Array.isArray(payload.lines) ? payload.lines : [];
      if (!lines.length) {
        const empty = document.createElement('span');
        empty.className = 'log-line';
        empty.textContent = 'No log entries yet';
        viewer.appendChild(empty);
        return;
      }
      for (const encodedLine of lines) {
        const line = decodeHtml(encodedLine);
        const row = document.createElement('div');
        row.className = 'log-line';
        const text = document.createElement('span');
        text.className = line.includes('ERROR')
          ? 'lvl-ERROR'
          : line.includes('WARNING')
            ? 'lvl-WARNING'
            : line.includes('INFO')
              ? 'lvl-INFO'
              : '';
        text.textContent = line;
        row.appendChild(text);
        viewer.appendChild(row);
      }
    } catch (_error) {
      viewer.textContent = 'Logs unavailable';
    }
  };
})();
</script>
""".replace("__ENTROLY_CONTROL_TOKEN__", json.dumps(CONTROL_TOKEN))


def _harden_controls_html(source: str) -> tuple[str, list[str]]:
    hardened, error = _inject_once(
        source,
        "</body>",
        _SAFE_CONTROLS_SCRIPT,
        label="controls-security-script",
    )
    return hardened, [error] if error else []


def _request_client_is_loopback(handler: Any) -> bool:
    try:
        address = ipaddress.ip_address(str(handler.client_address[0]))
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    return address.is_loopback


def _request_host_is_dashboard(handler: Any) -> bool:
    raw_host = handler.headers.get("Host", "")
    if not raw_host or len(raw_host) > 512 or any(ord(character) < 33 for character in raw_host):
        return False
    try:
        parsed = urlsplit("//" + raw_host)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.username is None
        and parsed.password is None
        and hostname in _ALLOWED_HOSTS
        and port == int(handler.server.server_port)
    )


def _request_origin_is_dashboard(handler: Any) -> bool:
    origin = handler.headers.get("Origin")
    if not origin:
        return True
    if len(origin) > 1024 or any(ord(character) < 33 for character in origin):
        return False
    try:
        parsed = urlsplit(origin)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        return (
            parsed.scheme == "http"
            and hostname in _ALLOWED_HOSTS
            and parsed.port == int(handler.server.server_port)
            and parsed.username is None
            and parsed.password is None
            and not parsed.path.rstrip("/")
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def _request_fetch_site_is_safe(handler: Any) -> bool:
    value = handler.headers.get("Sec-Fetch-Site")
    return not value or value.casefold() in {"same-origin", "same-site", "none"}


def _request_token_is_valid(handler: Any) -> bool:
    supplied = handler.headers.get("X-Entroly-Control-Token", "")
    return (
        isinstance(supplied, str)
        and len(supplied) <= _MAX_TOKEN_CHARS
        and hmac.compare_digest(supplied, CONTROL_TOKEN)
    )


_ORIGINAL_HANDLER = _dashboard.DashboardHandler
_ORIGINAL_DO_POST = _ORIGINAL_HANDLER.do_POST
_ORIGINAL_SECURITY_HEADERS = _ORIGINAL_HANDLER._send_security_headers


class SafeDashboardHandler(_ORIGINAL_HANDLER):
    """Loopback-only, capability-protected dashboard request handler."""

    server_version = "EntrolyDashboard"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(
            _env_float(
                "ENTROLY_DASHBOARD_SOCKET_TIMEOUT",
                _DEFAULT_SOCKET_TIMEOUT_S,
                1.0,
                30.0,
            )
        )

    def _send_security_headers(self) -> None:
        _ORIGINAL_SECURITY_HEADERS(self)
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("X-Permitted-Cross-Domain-Policies", "none")
        self.send_header("Cache-Control", "no-store, max-age=0")

    def _trusted_request_context(self) -> bool:
        return (
            _request_client_is_loopback(self)
            and _request_host_is_dashboard(self)
            and _request_origin_is_dashboard(self)
            and _request_fetch_site_is_safe(self)
        )

    def _reject(self, status: int, error: str) -> None:
        self._send_json(status, {"ok": False, "error": error}, cors=False)

    def _send_json(self, status: int, payload: dict, *, cors: bool = True) -> None:
        path = urlsplit(self.path).path
        body = _display_safe(payload) if path in _DISPLAY_SAFE_PATHS else payload
        origin = self.headers.get("Origin")
        cors_origin = origin if cors and origin and _request_origin_is_dashboard(self) else None
        self._respond(
            status,
            "application/json; charset=utf-8",
            json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"),
            no_cache=True,
            cors_origin=cors_origin,
        )

    def do_GET(self) -> None:
        if len(self.path) > _MAX_REQUEST_TARGET_CHARS:
            self._reject(414, "request target too long")
            return
        if not self._trusted_request_context():
            self._reject(403, "untrusted dashboard request context")
            return
        super().do_GET()

    def do_POST(self) -> None:
        if len(self.path) > _MAX_REQUEST_TARGET_CHARS:
            self._reject(414, "request target too long")
            return
        if not self._trusted_request_context():
            self._reject(403, "untrusted dashboard request context")
            return
        if not _request_token_is_valid(self):
            self._reject(403, "invalid control capability")
            return
        if self.headers.get("Transfer-Encoding"):
            self._reject(400, "chunked control requests are unsupported")
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
        if content_type != "application/json":
            self._reject(415, "control requests require application/json")
            return
        _ORIGINAL_DO_POST(self)

    def do_OPTIONS(self) -> None:
        if not self._trusted_request_context():
            self._reject(403, "untrusted dashboard request context")
            return
        self.send_response(204)
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Entroly-Control-Token",
        )
        self._send_security_headers()
        self.end_headers()


class BoundedThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server with a hard cap and immediate overload rejection."""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    request_queue_size = 64

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[SafeDashboardHandler],
        *,
        max_workers: int | None = None,
    ) -> None:
        if max_workers is None:
            workers = _env_int(
                "ENTROLY_DASHBOARD_MAX_CONNECTIONS",
                _DEFAULT_MAX_WORKERS,
                4,
                128,
            )
        else:
            if isinstance(max_workers, bool):
                raise ValueError("max_workers must be an integer between 1 and 128")
            try:
                workers = int(max_workers)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_workers must be an integer between 1 and 128") from exc
            if not 1 <= workers <= 128:
                raise ValueError("max_workers must be an integer between 1 and 128")
        self.max_workers = workers
        self._worker_slots = threading.BoundedSemaphore(workers)
        super().__init__(server_address, request_handler_class)

    @staticmethod
    def _reject_over_capacity(request: Any) -> None:
        body = b'{"error":"dashboard_busy","retry_after_s":1}'
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json\r\n"
            b"Cache-Control: no-store\r\n"
            b"Connection: close\r\n"
            b"Retry-After: 1\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        try:
            request.settimeout(1.0)
            request.sendall(response)
        except OSError:
            pass

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._worker_slots.acquire(blocking=False):
            self._reject_over_capacity(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()


def _register_lite_daemon(
    engine: Any,
    port: int,
    proxy_runtime: Any | None,
) -> Any:
    from entroly.daemon import (
        EntrolyDaemon,
        EntrolyDaemonState,
        RepoState,
        _install_log_buffer,
        _register_control_api,
        get_daemon,
    )

    _install_log_buffer()
    existing = get_daemon()
    if existing is not None:
        if proxy_runtime is not None:
            existing._proxy_runtime = proxy_runtime
            existing._proxy_config = getattr(proxy_runtime, "config", None)
        return existing

    lite = EntrolyDaemon.__new__(EntrolyDaemon)
    lite.state = EntrolyDaemonState()
    lite.state.status = "running"
    lite.state.started_at = time.time()
    lite.state.dashboard.running = True
    lite.state.dashboard.port = port
    lite.state.proxy.running = True
    lite.state.proxy.port = 9377
    lite._engine = engine
    lite._proxy_server = None
    lite._dashboard_server = None
    lite._workers = {}
    lite._shutdown = threading.Event()
    lite._lock = threading.Lock()
    lite._host = "127.0.0.1"
    lite._enable_proxy = True
    lite._enable_mcp = False
    lite._repo_paths = [os.getcwd()]
    lite._proxy_runtime = proxy_runtime
    lite._proxy_config = getattr(proxy_runtime, "config", None)

    try:
        stats = engine._rust.stats() if hasattr(engine, "_rust") else {}
        session = stats.get("session", {})
        lite.state.repos.append(
            RepoState(
                path=os.getcwd(),
                watching=True,
                indexed_files=session.get("total_fragments", 0),
                total_tokens=session.get("total_tokens_tracked", 0),
                last_sync=time.time(),
            )
        )
    except Exception:
        pass
    _register_control_api(lite)
    return lite


def start_dashboard(
    engine: Any = None,
    port: int = 9378,
    daemon: bool = True,
    proxy_runtime: Any | None = None,
) -> BoundedThreadingHTTPServer:
    """Start a bounded, capability-protected dashboard on IPv4 loopback."""
    if _HARDENING_ERRORS:
        raise RuntimeError(
            "dashboard security contract mismatch: " + "; ".join(_HARDENING_ERRORS)
        )
    if isinstance(port, bool):
        raise ValueError("dashboard port must be an integer")
    try:
        port_value = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError("dashboard port must be an integer") from exc
    if not 0 <= port_value <= 65535:
        raise ValueError("dashboard port must be between 0 and 65535")

    _dashboard._engine = engine
    state_owner = _register_lite_daemon(engine, port_value, proxy_runtime)
    server = BoundedThreadingHTTPServer(
        ("127.0.0.1", port_value),
        SafeDashboardHandler,
    )
    actual_port = int(server.server_port)
    try:
        state_owner.state.dashboard.port = actual_port
        state_owner.state.dashboard.running = True
    except Exception:
        pass
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=daemon,
        name="entroly-dashboard",
    )
    thread.start()
    logger.info("Dashboard live at http://localhost:%d", actual_port)
    return server


def install_dashboard_security() -> bool:
    """Install dashboard hardening exactly once and return whether it is usable."""
    global _INSTALLED, _HARDENING_ERRORS
    if _INSTALLED:
        return not _HARDENING_ERRORS

    controls_html, controls_errors = _harden_controls_html(_controls.CONTROLS_HTML)
    _HARDENING_ERRORS = tuple(controls_errors)
    if _HARDENING_ERRORS:
        logger.critical(
            "Dashboard hardening disabled startup because UI contracts drifted: %s",
            "; ".join(_HARDENING_ERRORS),
        )
    else:
        _controls.CONTROLS_HTML = controls_html

    _dashboard.DashboardHandler = SafeDashboardHandler
    _dashboard.start_dashboard = start_dashboard
    package = sys.modules.get("entroly")
    if package is not None and hasattr(package, "start_dashboard"):
        setattr(package, "start_dashboard", start_dashboard)
    _INSTALLED = True
    return not _HARDENING_ERRORS


install_dashboard_security()

__all__ = [
    "BoundedThreadingHTTPServer",
    "CONTROL_TOKEN",
    "SafeDashboardHandler",
    "control_token",
    "install_dashboard_security",
    "start_dashboard",
]
