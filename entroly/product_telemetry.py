"""Explicit-consent, content-blind product health telemetry.

This module exists to answer two narrow product questions:

* are consenting installations reaching useful Entroly surfaces;
* which coarse command/surface error categories are they encountering; and
* are those installations observing a positive, locally measured reduction?

It is deliberately unsuitable for prompts, source code, file paths, model
inputs, exception messages, tracebacks, hostnames, usernames, credentials,
exact token counts, exact costs, or model identifiers.
The event schema is a closed allowlist and telemetry is disabled until the
operator runs ``entroly telemetry on``. Air-gap mode and the hard-disable
environment flag always win over stored consent.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import os
import platform
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "entroly.product-telemetry.v1"
BATCH_SCHEMA_VERSION = "entroly.product-telemetry-batch.v1"
DELETE_SCHEMA_VERSION = "entroly.product-telemetry-delete.v1"
CONSENT_VERSION = 1
DEFAULT_RETENTION_DAYS = 14
MAX_QUEUE_EVENTS = 200
MAX_BATCH_EVENTS = 20
MAX_BATCH_BYTES = 64 * 1024

_CLI_COMMANDS = frozenset({
    "attach", "audit", "autotune", "batch", "benchmark", "cache",
    "capabilities", "clean", "compile", "compress", "config",
    "context-commit", "daemon", "dashboard", "demo", "digest", "docs",
    "doctor", "drift", "explain", "export", "feedback", "finetune", "go",
    "health", "import", "ingest", "init", "learn", "migrate", "optimize",
    "perf", "profile", "proof", "proxy", "ravs", "receipt", "recover",
    "role", "search", "select", "serve", "share", "simulate", "status",
    "sync", "telemetry", "unwrap", "value", "verify", "verify-claims",
    "verify-code", "witness", "wrap",
})
_SURFACES = frozenset({
    "cli", "compression_mcp", "mcp", "proxy", "repository_mcp",
    "sdk_compress", "sdk_messages",
})
_RESULTS = frozenset({"success", "error", "interrupted"})
_DURATION_BUCKETS = frozenset({"lt_100ms", "lt_1s", "lt_10s", "lt_60s", "gte_60s"})
_TOKEN_SAVINGS_BUCKETS = frozenset({
    "none", "1_99", "100_999", "1k_9k", "10k_99k", "100k_plus",
})
_REDUCTION_PERCENT_BUCKETS = frozenset({
    "none", "lt_10", "10_29", "30_49", "50_69", "70_89", "90_plus",
})
_MEASUREMENT_SCOPES = frozenset({"local_estimate", "provider_bound_estimate"})
_COST_EVIDENCE = frozenset({"not_available", "modeled_positive"})
_ERROR_TYPES = frozenset({
    "AssertionError", "ConnectionError", "ImportError", "LookupError",
    "MemoryError", "OSError", "PermissionError", "RuntimeError",
    "TimeoutError", "TypeError", "ValueError", "OtherError",
})
_EVENT_PROPERTIES = {
    "activation": frozenset({"surface"}),
    "command": frozenset({"command", "result", "duration_bucket", "error_type"}),
    "optimization_outcome": frozenset({
        "surface", "measurement_scope", "tokens_saved_bucket",
        "reduction_percent_bucket", "cost_evidence",
    }),
    "surface_started": frozenset({"surface"}),
    "surface_error": frozenset({"surface", "error_type"}),
}
_REQUIRED_EVENT_PROPERTIES = {
    "activation": frozenset({"surface"}),
    "command": frozenset({"command", "result", "duration_bucket"}),
    "optimization_outcome": _EVENT_PROPERTIES["optimization_outcome"],
    "surface_started": frozenset({"surface"}),
    "surface_error": frozenset({"surface", "error_type"}),
}
_HEX_24 = re.compile(r"^[0-9a-f]{24}$")
_UUID_HEX = re.compile(r"^[0-9a-f]{32}$")
_VERSION = re.compile(r"^[0-9A-Za-z.+_-]{1,32}$")
_LOCK = threading.RLock()
_SEEN_DAILY: set[str] = set()
_LEGACY_SEED_KEY = "anonymous" + "_seed"


def _state_dir() -> Path:
    configured = os.environ.get("ENTROLY_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".entroly"


def _config_path() -> Path:
    return _state_dir() / "telemetry.json"


def _queue_path() -> Path:
    return _state_dir() / "product-telemetry.jsonl"


def _markers_path() -> Path:
    return _state_dir() / "product-telemetry-markers.json"


def _status_path() -> Path:
    return _state_dir() / "product-telemetry-status.json"


def _lock_path() -> Path:
    return _state_dir() / "product-telemetry.lock"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _hard_disabled() -> bool:
    testing_override = os.environ.get("ENTROLY_TELEMETRY_TESTING", "0") == "1"
    if os.environ.get("PYTEST_CURRENT_TEST") and not testing_override:
        return True
    include_ci = os.environ.get("ENTROLY_TELEMETRY_INCLUDE_CI", "0") == "1"
    if os.environ.get("CI", "").strip().casefold() in {"1", "true", "yes"} and not include_ci:
        return True
    raw = os.environ.get("ENTROLY_DISABLE_TELEMETRY", "0").strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    try:
        from .air_gap import air_gap_enabled

        return bool(air_gap_enabled())
    except Exception:
        return os.environ.get("ENTROLY_AIR_GAP", "0") == "1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    os.replace(temp, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


@contextlib.contextmanager
def _process_lock(timeout: float = 0.5) -> Iterator[None]:
    """Bound concurrent queue updates; telemetry drops instead of blocking."""
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a+b") as handle:
        deadline = time.monotonic() + max(0.01, timeout)
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, PermissionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("telemetry queue is busy") from None
                time.sleep(0.005)
        try:
            yield
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, PermissionError):
                pass


def validate_endpoint(value: str) -> str:
    """Accept HTTPS collectors or explicit loopback HTTP development URLs."""
    parsed = urllib.parse.urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("telemetry endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("telemetry endpoint cannot contain credentials, query, or fragment")
    if parsed.scheme == "http":
        hostname = parsed.hostname.casefold()
        loopback = hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise ValueError("remote telemetry endpoints must use HTTPS")
    return urllib.parse.urlunsplit(parsed)


def _load_config() -> dict[str, Any]:
    config = _load_json(_config_path())
    if config.get("enabled") is not True:
        return {}
    if config.get("consent_version") != CONSENT_VERSION:
        return {}
    seed = config.get("pseudonym_seed") or config.get(_LEGACY_SEED_KEY)
    if not isinstance(seed, str) or not _UUID_HEX.fullmatch(seed):
        return {}
    return config


def _configured_endpoint(config: dict[str, Any]) -> str | None:
    # The destination is part of the recorded consent. An environment variable
    # must not turn a previously local-only preference into an outbound call.
    raw = str(config.get("endpoint", "")).strip()
    if not raw:
        return None
    try:
        return validate_endpoint(raw)
    except ValueError:
        return None


def enable(*, endpoint: str | None = None, error_events: bool = True) -> dict[str, Any]:
    """Persist explicit consent. No environment variable can silently opt in."""
    if _hard_disabled():
        raise RuntimeError("telemetry is blocked by air-gap or hard-disable policy")
    normalized_endpoint = validate_endpoint(endpoint) if endpoint else None
    previous = _load_json(_config_path())
    seed = previous.get("pseudonym_seed") or previous.get(_LEGACY_SEED_KEY)
    if not isinstance(seed, str) or not _UUID_HEX.fullmatch(seed):
        seed = uuid.uuid4().hex
    config: dict[str, Any] = {
        "enabled": True,
        "consent_version": CONSENT_VERSION,
        "opted_in_on": _today(),
        "pseudonym_seed": seed,
        "error_events": bool(error_events),
    }
    if normalized_endpoint:
        config["endpoint"] = normalized_endpoint
    _atomic_json(_config_path(), config)
    capture("activation", {"surface": "cli"}, once_per_day=False)
    return status()


def disable_and_purge() -> dict[str, Any]:
    """Withdraw consent and delete the local queue and pseudonymous identity."""
    config = _load_config()
    remote_deletion = _request_remote_deletion(config) if config else "not_configured"
    removed: list[str] = []
    for path in (_queue_path(), _markers_path(), _status_path(), _config_path()):
        try:
            path.unlink()
            removed.append(path.name)
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return {
        "enabled": False,
        "purged": sorted(removed),
        "remote_deletion": remote_deletion,
    }


def is_enabled() -> bool:
    return not _hard_disabled() and bool(_load_config())


def _installation_id(config: dict[str, Any], day: str) -> str:
    # A monthly pseudonym supports active-install counts without a permanent
    # cross-release identity. The seed never leaves the machine.
    month = day[:7]
    value = f"{config['pseudonym_seed']}:{month}:{CONSENT_VERSION}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _recent_installation_ids(config: dict[str, Any]) -> list[str]:
    today = datetime.now(timezone.utc).date()
    months: list[str] = []
    year, month = today.year, today.month
    for _ in range(4):
        months.append(f"{year:04d}-{month:02d}-01")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return [_installation_id(config, value) for value in months]


def _platform_family() -> str:
    value = platform.system().casefold()
    if value == "darwin":
        return "macos"
    if value in {"linux", "windows"}:
        return value
    return "other"


def _package_version() -> str:
    try:
        from . import __version__

        value = str(__version__)
    except Exception:
        value = "unknown"
    return value if _VERSION.fullmatch(value) else "unknown"


def safe_error_type(error: BaseException | str | None) -> str:
    if isinstance(error, BaseException):
        name = type(error).__name__
    elif isinstance(error, str):
        name = error
    else:
        name = "OtherError"
    return name if name in _ERROR_TYPES else "OtherError"


def duration_bucket(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value < 0.1:
        return "lt_100ms"
    if value < 1.0:
        return "lt_1s"
    if value < 10.0:
        return "lt_10s"
    if value < 60.0:
        return "lt_60s"
    return "gte_60s"


def tokens_saved_bucket(tokens_saved: int) -> str:
    """Return a non-sensitive bucket instead of an exact workload size."""
    try:
        value = max(0, int(tokens_saved))
    except (TypeError, ValueError, OverflowError):
        value = 0
    if value == 0:
        return "none"
    if value < 100:
        return "1_99"
    if value < 1_000:
        return "100_999"
    if value < 10_000:
        return "1k_9k"
    if value < 100_000:
        return "10k_99k"
    return "100k_plus"


def reduction_percent_bucket(before_tokens: int, after_tokens: int) -> str:
    """Bucket a before/after reduction without serializing either count."""
    try:
        before = max(0, int(before_tokens))
        after = max(0, int(after_tokens))
    except (TypeError, ValueError, OverflowError):
        return "none"
    saved = max(0, before - after)
    if before == 0 or saved == 0:
        return "none"
    percent = min(100, (saved * 100) // before)
    if percent < 10:
        return "lt_10"
    if percent < 30:
        return "10_29"
    if percent < 50:
        return "30_49"
    if percent < 70:
        return "50_69"
    if percent < 90:
        return "70_89"
    return "90_plus"


def _sanitize_properties(event_name: str, raw: dict[str, Any]) -> dict[str, str]:
    allowed = _EVENT_PROPERTIES.get(event_name, frozenset())
    result: dict[str, str] = {}
    for key in allowed:
        if key not in raw:
            continue
        value = raw.get(key)
        if key == "command":
            result[key] = str(value) if value in _CLI_COMMANDS else "other"
        elif key == "surface":
            result[key] = str(value) if value in _SURFACES else "cli"
        elif key == "result":
            result[key] = str(value) if value in _RESULTS else "error"
        elif key == "duration_bucket":
            result[key] = str(value) if value in _DURATION_BUCKETS else "gte_60s"
        elif key == "error_type":
            result[key] = safe_error_type(value if isinstance(value, str) else None)
        elif key == "tokens_saved_bucket":
            result[key] = str(value) if value in _TOKEN_SAVINGS_BUCKETS else "none"
        elif key == "reduction_percent_bucket":
            result[key] = str(value) if value in _REDUCTION_PERCENT_BUCKETS else "none"
        elif key == "measurement_scope":
            result[key] = str(value) if value in _MEASUREMENT_SCOPES else "local_estimate"
        elif key == "cost_evidence":
            result[key] = str(value) if value in _COST_EVIDENCE else "not_available"
    return result


def sanitize_public_event(value: Any, *, strict: bool = False) -> dict[str, Any] | None:
    """Validate the public wire schema; arbitrary properties never survive."""
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        return None
    event_name = value.get("event_name")
    if event_name not in _EVENT_PROPERTIES:
        return None
    event_id = value.get("event_id")
    installation_id = value.get("installation_id")
    occurred_on = value.get("occurred_on")
    if not isinstance(event_id, str) or not _UUID_HEX.fullmatch(event_id):
        return None
    if not isinstance(installation_id, str) or not _HEX_24.fullmatch(installation_id):
        return None
    try:
        date.fromisoformat(str(occurred_on))
    except ValueError:
        return None
    version = str(value.get("version", "unknown"))
    python_version = str(value.get("python", "unknown"))
    platform_name = str(value.get("platform", "other"))
    if not _VERSION.fullmatch(version):
        version = "unknown"
    if not re.fullmatch(r"(?:[0-9]{1,2}\.[0-9]{1,2}|unknown)", python_version):
        python_version = "unknown"
    if platform_name not in {"linux", "macos", "windows", "other"}:
        platform_name = "other"
    raw_properties = value.get("properties")
    if not isinstance(raw_properties, dict):
        raw_properties = {}
    properties = _sanitize_properties(str(event_name), raw_properties)
    if not _REQUIRED_EVENT_PROPERTIES[str(event_name)].issubset(properties):
        return None
    sanitized = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "occurred_on": str(occurred_on),
        "installation_id": installation_id,
        "event_name": str(event_name),
        "version": version,
        "platform": platform_name,
        "python": python_version,
        "properties": properties,
    }
    if strict and sanitized != value:
        return None
    return sanitized


def _read_queue() -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=DEFAULT_RETENTION_DAYS)
    rows: list[dict[str, Any]] = []
    try:
        lines = _queue_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            event = sanitize_public_event(json.loads(line), strict=True)
            if event and date.fromisoformat(event["occurred_on"]) >= cutoff:
                rows.append(event)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return rows[-MAX_QUEUE_EVENTS:]


def _write_queue(events: list[dict[str, Any]]) -> None:
    path = _queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    body = "".join(
        json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        for event in events[-MAX_QUEUE_EVENTS:]
    )
    temp.write_text(body, encoding="utf-8")
    try:
        os.chmod(temp, 0o600)
    except OSError:
        pass
    os.replace(temp, path)


def _claim_daily_marker(marker: str) -> bool:
    day = _today()
    markers = _load_json(_markers_path())
    if markers.get(marker) == day:
        return False
    markers = {
        str(key): str(value)
        for key, value in markers.items()
        if isinstance(key, str) and isinstance(value, str) and value >= day[:7]
    }
    markers[marker] = day
    _atomic_json(_markers_path(), markers)
    return True


def capture(
    event_name: str,
    properties: dict[str, Any] | None = None,
    *,
    once_per_day: bool = False,
) -> bool:
    """Queue one allowlisted event. Every failure is a silent no-op."""
    if not is_enabled() or event_name not in _EVENT_PROPERTIES:
        return False
    config = _load_config()
    if event_name == "surface_error" and config.get("error_events") is not True:
        return False
    if (
        event_name == "command"
        and config.get("error_events") is not True
        and properties
        and properties.get("result") == "error"
    ):
        properties = {key: value for key, value in properties.items() if key != "error_type"}
    try:
        sanitized_properties = _sanitize_properties(event_name, properties or {})
        marker = f"{event_name}:{json.dumps(sanitized_properties, sort_keys=True)}"
        daily_key = f"{_today()}:{marker}"
        if once_per_day and daily_key in _SEEN_DAILY:
            return False
        with _process_lock():
            if once_per_day and not _claim_daily_marker(marker):
                _SEEN_DAILY.add(daily_key)
                return False
            if once_per_day:
                _SEEN_DAILY.add(daily_key)
            day = _today()
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_id": uuid.uuid4().hex,
                "occurred_on": day,
                "installation_id": _installation_id(config, day),
                "event_name": event_name,
                "version": _package_version(),
                "platform": _platform_family(),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "properties": sanitized_properties,
            }
            sanitized = sanitize_public_event(event, strict=True)
            if sanitized is None:
                return False
            events = _read_queue()
            events.append(sanitized)
            _write_queue(events)
        return True
    except Exception:
        return False


def capture_cli_result(
    command: str | None,
    *,
    result: str,
    elapsed_seconds: float,
    error: BaseException | None = None,
) -> bool:
    properties: dict[str, Any] = {
        "command": command or "other",
        "result": result,
        "duration_bucket": duration_bucket(elapsed_seconds),
    }
    if result == "error":
        properties["error_type"] = safe_error_type(error)
    return capture("command", properties, once_per_day=True)


def capture_surface_started(surface: str) -> bool:
    return capture("surface_started", {"surface": surface}, once_per_day=True)


def capture_surface_error(surface: str, error: BaseException) -> bool:
    return capture(
        "surface_error",
        {"surface": surface, "error_type": safe_error_type(error)},
        once_per_day=True,
    )


def capture_optimization_outcome(
    surface: str,
    *,
    before_tokens: int,
    after_tokens: int,
    measurement_scope: str = "local_estimate",
    cost_evidence: str = "not_available",
) -> bool:
    """Record a daily coarse value signal without usage volume or content.

    The same surface/bucket combination is emitted at most once per UTC day.
    ``modeled_positive`` means a provider-bound reduction had an explicitly
    priced model; it is still not a provider invoice or observed charge.
    """
    try:
        saved = max(0, int(before_tokens) - int(after_tokens))
    except (TypeError, ValueError, OverflowError):
        saved = 0
    return capture(
        "optimization_outcome",
        {
            "surface": surface,
            "measurement_scope": measurement_scope,
            "tokens_saved_bucket": tokens_saved_bucket(saved),
            "reduction_percent_bucket": reduction_percent_bucket(
                before_tokens, after_tokens
            ),
            "cost_evidence": cost_evidence,
        },
        once_per_day=True,
    )


def _opener() -> urllib.request.OpenerDirector:
    trust_proxy = os.environ.get(
        "ENTROLY_TELEMETRY_TRUST_PROXY_ENV", "0"
    ).strip().casefold() in {"1", "true", "yes", "on"}
    return urllib.request.build_opener(
        urllib.request.ProxyHandler() if trust_proxy else urllib.request.ProxyHandler({})
    )


def _request_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": f"entroly-product-health/{_package_version()}",
    }
    token = os.environ.get("ENTROLY_TELEMETRY_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_timeout() -> float:
    raw = os.environ.get("ENTROLY_TELEMETRY_TIMEOUT_SECONDS", "1.0")
    try:
        return max(0.1, min(float(raw), 3.0))
    except ValueError:
        return 1.0


def _request_remote_deletion(config: dict[str, Any]) -> str:
    endpoint = _configured_endpoint(config)
    if endpoint is None:
        return "not_configured"
    if _hard_disabled():
        return "blocked"
    body = json.dumps(
        {
            "schema_version": DELETE_SCHEMA_VERSION,
            "installation_ids": _recent_installation_ids(config),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers=_request_headers(),
        method="DELETE",
    )
    try:
        with _opener().open(request, _request_timeout()) as response:
            status_code = int(getattr(response, "status", 0) or response.getcode())
            return "deleted" if 200 <= status_code < 300 else "error"
    except Exception:
        return "error"


def flush(
    *, max_events: int = MAX_BATCH_EVENTS, force: bool = False
) -> dict[str, Any]:
    """Upload a bounded batch after consent; never raises or drops on failure.

    Automatic attempts happen at most once per UTC day. Operators can request
    an explicit retry with ``force=True`` via ``entroly telemetry flush``.
    """
    if not is_enabled():
        return {"status": "disabled", "sent": 0}
    config = _load_config()
    endpoint = _configured_endpoint(config)
    if endpoint is None:
        return {"status": "not_configured", "sent": 0}
    prior_status = _load_json(_status_path())
    if not force and prior_status.get("last_attempt_on") == _today():
        return {"status": "deferred", "sent": 0}
    try:
        with _process_lock():
            events = _read_queue()
        batch = events[: max(1, min(int(max_events), MAX_BATCH_EVENTS))]
        if not batch:
            return {"status": "empty", "sent": 0}
        body = json.dumps(
            {"schema_version": BATCH_SCHEMA_VERSION, "events": batch},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > MAX_BATCH_BYTES:
            return {"status": "batch_too_large", "sent": 0}
        request = urllib.request.Request(
            endpoint, data=body, headers=_request_headers(), method="POST"
        )
        with _opener().open(request, _request_timeout()) as response:
            status_code = int(getattr(response, "status", 0) or response.getcode())
            if not 200 <= status_code < 300:
                raise urllib.error.HTTPError(
                    endpoint, status_code, "telemetry collector rejected batch", {}, None
                )
        sent_ids = {event["event_id"] for event in batch}
        with _process_lock():
            remaining = [event for event in _read_queue() if event["event_id"] not in sent_ids]
            _write_queue(remaining)
        _atomic_json(
            _status_path(),
            {"last_attempt_on": _today(), "last_result": "success", "sent": len(batch)},
        )
        return {"status": "sent", "sent": len(batch)}
    except Exception as error:
        # Only the broad error category is kept locally. Messages, URLs,
        # response bodies, and tracebacks can contain sensitive material.
        try:
            _atomic_json(
                _status_path(),
                {
                    "last_attempt_on": _today(),
                    "last_result": "error",
                    "error_type": safe_error_type(error),
                },
            )
        except Exception:
            pass
        return {"status": "error", "sent": 0, "error_type": safe_error_type(error)}


def flush_async() -> None:
    if not is_enabled() or _configured_endpoint(_load_config()) is None:
        return
    threading.Thread(target=flush, name="entroly-product-telemetry", daemon=True).start()


def preview() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "events": sorted(_EVENT_PROPERTIES),
        "allowed_properties": {
            key: sorted(value) for key, value in sorted(_EVENT_PROPERTIES.items())
        },
        "never_collected": [
            "prompts", "source_code", "file_paths", "filenames", "model_inputs",
            "model_outputs", "exception_messages", "tracebacks", "hostnames",
            "usernames", "environment_values", "credentials", "ip_addresses",
            "exact_token_counts", "exact_costs", "model_identifiers",
        ],
        "identifier": "random monthly-rotating pseudonym; local seed never uploaded",
        "frequency_protection": (
            "command, surface, error, and value categories are deduplicated per UTC day"
        ),
        "retention_days_local_queue": DEFAULT_RETENTION_DAYS,
        "enabled_by_default": False,
    }


def status() -> dict[str, Any]:
    config = _load_config()
    enabled = bool(config) and not _hard_disabled()
    endpoint = _configured_endpoint(config) if enabled else None
    queued = 0
    if _queue_path().exists():
        try:
            with _process_lock():
                queued = len(_read_queue())
        except Exception:
            queued = 0
    result: dict[str, Any] = {
        "enabled": enabled,
        "hard_disabled": _hard_disabled(),
        "consent_version": config.get("consent_version") if enabled else None,
        "error_events": bool(config.get("error_events")) if enabled else False,
        "upload_configured": endpoint is not None,
        "endpoint_origin": None,
        "queued_events": queued,
        "local_retention_days": DEFAULT_RETENTION_DAYS,
    }
    if endpoint:
        parsed = urllib.parse.urlsplit(endpoint)
        result["endpoint_origin"] = f"{parsed.scheme}://{parsed.netloc}"
    upload_status = _load_json(_status_path())
    if upload_status:
        result["last_upload"] = upload_status
    return result


__all__ = [
    "BATCH_SCHEMA_VERSION",
    "CONSENT_VERSION",
    "DELETE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "capture",
    "capture_cli_result",
    "capture_optimization_outcome",
    "capture_surface_error",
    "capture_surface_started",
    "disable_and_purge",
    "duration_bucket",
    "enable",
    "flush",
    "flush_async",
    "is_enabled",
    "preview",
    "reduction_percent_bucket",
    "safe_error_type",
    "sanitize_public_event",
    "status",
    "tokens_saved_bucket",
    "validate_endpoint",
]
