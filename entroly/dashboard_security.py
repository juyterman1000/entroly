"""Security boundary for Entroly's localhost dashboard and control API.

The dashboard renders data derived from repository paths, prompts, model output,
security findings, selection reasons, and exception messages. Those values are
not trusted merely because the listener is on loopback: a malicious repository
or poisoned telemetry record can become stored HTML/JavaScript in the browser,
and any script executing in the dashboard origin can invoke daemon controls.

This module installs four fail-closed controls without changing the public
``start_dashboard`` API:

* every known attacker-controlled ``innerHTML`` sink is escaped;
* state-changing routes require a per-process capability token;
* Host, Origin, client address, content type, and request target are validated;
* the HTTP server has bounded workers, bounded backlog, and socket timeouts.

It is imported during normal package initialization after ``dashboard.py`` and
``controls_html.py`` are available. If the embedded UI changes so an expected
rewrite no longer matches, dashboard startup is disabled rather than silently
serving a partially hardened control plane.
"""

from __future__ import annotations

import hmac
import html
import ipaddress
import logging
import math
import os
import secrets
import socket
import sys
import threading
import time
from http.server import HTTPServer
from socketserver import ThreadingMixIn
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from . import controls_html as _controls
from . import dashboard as _dashboard

logger = logging.getLogger("entroly.dashboard.security")

_DEFAULT_MAX_WORKERS = 24
_DEFAULT_SOCKET_TIMEOUT_S = 5.0
_MAX_REQUEST_TARGET_CHARS = 8192
_MAX_TOKEN_CHARS = 512
_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
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
    """Return the active token for trusted in-process automation and tests."""
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


def _replace_counted(
    text: str,
    old: str,
    new: str,
    *,
    label: str,
    expected: int = 1,
) -> tuple[str, str | None]:
    count = text.count(old)
    if count != expected:
        return text, f"{label}: expected {expected} occurrence(s), found {count}"
    return text.replace(old, new), None


def _apply_replacements(
    text: str,
    replacements: list[tuple[str, str, str, int]],
) -> tuple[str, list[str]]:
    errors: list[str] = []
    for label, old, new, expected in replacements:
        text, error = _replace_counted(
            text,
            old,
            new,
            label=label,
            expected=expected,
        )
        if error:
            errors.append(error)
    return text, errors


def _harden_dashboard_html(source: str) -> tuple[str, list[str]]:
    replacements = [
        (
            "dashboard-js-argument-encoder",
            """function escHtml(s){
  return String(s==null?'':s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));
}""",
            """function escHtml(s){
  return String(s==null?'':s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));
}
function jsArg(s){return escHtml(JSON.stringify(String(s==null?'':s)));}""",
            1,
        ),
        (
            "health-recommendation",
            "${h.top_recommendation?'<div class=\"health-rec\">💡 '+h.top_recommendation+'</div>':''}`;",
            "${h.top_recommendation?'<div class=\"health-rec\">💡 '+escHtml(h.top_recommendation)+'</div>':''}`;",
            1,
        ),
        (
            "security-finding",
            """panels+=findings.slice(0,4).map(f=>`<div class="finding"><span class="finding-sev ${(f.severity||'').toLowerCase()==='critical'?'crit':'high'}">${(f.severity||'?')[0]}</span><div><div class="finding-file">${f.file||f.source||'unknown'}${f.line?':'+f.line:''}</div><div class="finding-desc">${f.message||f.category||''}</div></div></div>`).join('');""",
            """panels+=findings.slice(0,4).map(f=>`<div class="finding"><span class="finding-sev ${(f.severity||'').toLowerCase()==='critical'?'crit':'high'}">${escHtml((f.severity||'?')[0])}</span><div><div class="finding-file">${escHtml(f.file||f.source||'unknown')}${f.line?':'+escHtml(f.line):''}</div><div class="finding-desc">${escHtml(f.message||f.category||'')}</div></div></div>`).join('');""",
            1,
        ),
        (
            "security-category",
            """panels+=Object.entries(cats).map(([k,v])=>`<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;"><span style="color:var(--dim);">${k}</span><span class="tag t-rose">${v}</span></div>`).join('');""",
            """panels+=Object.entries(cats).map(([k,v])=>`<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;"><span style="color:var(--dim);">${escHtml(k)}</span><span class="tag t-rose">${escHtml(v)}</span></div>`).join('');""",
            1,
        ),
        (
            "knapsack-source",
            "${(f.source||f.id||'').split(/[\\\\/]/).pop()}",
            "${escHtml((f.source||f.id||'').split(/[\\\\/]/).pop())}",
            2,
        ),
        (
            "knapsack-included-reason",
            "${(f.reason||'').slice(0,30)}",
            "${escHtml((f.reason||'').slice(0,30))}",
            1,
        ),
        (
            "knapsack-excluded-reason",
            "${(f.reason||'below threshold').slice(0,30)}",
            "${escHtml((f.reason||'below threshold').slice(0,30))}",
            1,
        ),
        (
            "request-model",
            "${r.model||'—'}",
            "${escHtml(r.model||'—')}",
            1,
        ),
        (
            "request-query",
            "${r.query||'—'}",
            "${escHtml(r.query||'—')}",
            1,
        ),
        (
            "dashboard-error-banner",
            """const list=items.slice(0,5).map(x=>`<div style="margin-top:6px;opacity:.85"><code style="background:rgba(239,68,68,.12);padding:1px 6px;border-radius:4px">${x.section||'?'}</code> ${x.type||'Error'}: ${(x.message||'').substring(0,200)}</div>`).join('');""",
            """const list=items.slice(0,5).map(x=>`<div style="margin-top:6px;opacity:.85"><code style="background:rgba(239,68,68,.12);padding:1px 6px;border-radius:4px">${escHtml(x.section||'?')}</code> ${escHtml(x.type||'Error')}: ${escHtml((x.message||'').substring(0,200))}</div>`).join('');""",
            1,
        ),
        (
            "health-grade",
            "<div class=\"grade\" style=\"color:${gc}\">${g}</div>",
            "<div class=\"grade\" style=\"color:${gc}\">${escHtml(g)}</div>",
            1,
        ),
        (
            "cogops-engine",
            "<div class=\"cache-kpi-val hv-violet\">${c.engine||'—'}</div>",
            "<div class=\"cache-kpi-val hv-violet\">${escHtml(c.engine||'—')}</div>",
            1,
        ),
        (
            "pricing-as-of",
            "'Rates as of '+((d.pricing||{}).as_of||'—')+' · '",
            "'Rates as of '+escHtml((d.pricing||{}).as_of||'—')+' · '",
            1,
        ),
        (
            "trend-status",
            "+'</span>'+status+'</span></div>'+",
            "+'</span>'+escHtml(status)+'</span></div>'+",
            1,
        ),
    ]
    return _apply_replacements(source, replacements)


def _harden_controls_html(source: str) -> tuple[str, list[str]]:
    token_meta = (
        '<meta name="entroly-control-token" content="'
        + html.escape(CONTROL_TOKEN, quote=True)
        + '">\n'
    )
    replacements = [
        (
            "control-token-meta",
            "</head>",
            token_meta + "</head>",
            1,
        ),
        (
            "control-js-safety-helpers",
            """<script>
function toast(msg,ok=true){""",
            """<script>
const ENTROLY_CONTROL_TOKEN=document.querySelector('meta[name="entroly-control-token"]').content;
function escHtml(s){return String(s==null?'':s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));}
function jsArg(s){return escHtml(JSON.stringify(String(s==null?'':s)));}
function toast(msg,ok=true){""",
            1,
        ),
        (
            "control-token-header",
            "headers:{'Content-Type':'application/json'}",
            "headers:{'Content-Type':'application/json','X-Entroly-Control-Token':ENTROLY_CONTROL_TOKEN}",
            1,
        ),
        (
            "control-repository-rendering",
            """function renderRepos(repos){if(!repos||!repos.length)return'<div style="color:var(--dim);font-size:13px;">No repos configured</div>';
return repos.map(r=>`<div class="repo-item"><span class="repo-icon">${r.watching?'&#128994;':'&#128308;'}</span>
<div class="repo-info"><div class="repo-path">${r.path}</div>
<div class="repo-meta">${r.indexed_files||0} files &middot; ${(r.total_tokens||0).toLocaleString()} tokens${r.last_sync?' &middot; synced '+new Date(r.last_sync*1000).toLocaleTimeString():''}</div></div>
<button class="btn" onclick="ctrlPost('/api/control/repos/reindex',{path:'${r.path.replace(/\\/g,'\\\\')}'})" style="flex-shrink:0;">Re-index</button></div>`).join('');}""",
            """function renderRepos(repos){if(!repos||!repos.length)return'<div style="color:var(--dim);font-size:13px;">No repos configured</div>';
return repos.map(r=>{const path=String(r.path||'');return `<div class="repo-item"><span class="repo-icon">${r.watching?'&#128994;':'&#128308;'}</span>
<div class="repo-info"><div class="repo-path">${escHtml(path)}</div>
<div class="repo-meta">${r.indexed_files||0} files &middot; ${(r.total_tokens||0).toLocaleString()} tokens${r.last_sync?' &middot; synced '+new Date(r.last_sync*1000).toLocaleTimeString():''}</div></div>
<button class="btn" onclick="ctrlPost('/api/control/repos/reindex',{path:${jsArg(path)}})" style="flex-shrink:0;">Re-index</button></div>`;}).join('');}""",
            1,
        ),
        (
            "control-context-source",
            "<span style=\"color:var(--emerald);\">&#10003; ${src}</span>",
            "<span style=\"color:var(--emerald);\">&#10003; ${escHtml(src)}</span>",
            1,
        ),
        (
            "control-log-escaping",
            "l.replace(/</g,'&lt;')",
            "escHtml(l)",
            1,
        ),
        (
            "control-daemon-status",
            "'Status: <b>'+s.status+'</b><br>Uptime: '",
            "'Status: <b>'+escHtml(s.status)+'</b><br>Uptime: '",
            1,
        ),
    ]
    return _apply_replacements(source, replacements)


def _request_client_is_loopback(handler: Any) -> bool:
    try:
        address = ipaddress.ip_address(str(handler.client_address[0]))
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    return address.is_loopback


def _request_host_is_dashboard(handler: Any) -> bool:
    raw_host = handler.headers.get("Host", "")
    if not raw_host or len(raw_host) > 512 or any(ord(c) < 33 for c in raw_host):
        return False
    try:
        parsed = urlsplit("//" + raw_host)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        port = parsed.port
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if hostname not in _ALLOWED_HOSTS:
        return False
    return port == int(handler.server.server_port)


def _request_origin_is_dashboard(handler: Any) -> bool:
    origin = handler.headers.get("Origin")
    if not origin:
        return True
    if len(origin) > 1024 or any(ord(c) < 33 for c in origin):
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
    if not value:
        return True
    return value.casefold() in {"same-origin", "same-site", "none"}


def _request_token_is_valid(handler: Any) -> bool:
    supplied = handler.headers.get("X-Entroly-Control-Token", "")
    if not isinstance(supplied, str) or len(supplied) > _MAX_TOKEN_CHARS:
        return False
    return hmac.compare_digest(supplied, CONTROL_TOKEN)


_ORIGINAL_HANDLER = _dashboard.DashboardHandler
_ORIGINAL_DO_POST = _ORIGINAL_HANDLER.do_POST
_ORIGINAL_SECURITY_HEADERS = _ORIGINAL_HANDLER._send_security_headers


class SafeDashboardHandler(_ORIGINAL_HANDLER):
    """Loopback-only, capability-protected dashboard request handler."""

    server_version = "EntrolyDashboard"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        timeout = _env_float(
            "ENTROLY_DASHBOARD_SOCKET_TIMEOUT",
            _DEFAULT_SOCKET_TIMEOUT_S,
            1.0,
            30.0,
        )
        self.connection.settimeout(timeout)

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
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
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
        workers = max_workers or _env_int(
            "ENTROLY_DASHBOARD_MAX_CONNECTIONS",
            _DEFAULT_MAX_WORKERS,
            4,
            128,
        )
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

    dashboard_html, dashboard_errors = _harden_dashboard_html(
        _dashboard.DASHBOARD_HTML
    )
    controls_html, controls_errors = _harden_controls_html(
        _controls.CONTROLS_HTML
    )
    _HARDENING_ERRORS = tuple(dashboard_errors + controls_errors)
    if _HARDENING_ERRORS:
        logger.critical(
            "Dashboard hardening disabled startup because UI contracts drifted: %s",
            "; ".join(_HARDENING_ERRORS),
        )
    else:
        _dashboard.DASHBOARD_HTML = dashboard_html
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
