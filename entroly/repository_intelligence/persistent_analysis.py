"""Content-addressed persistence for deterministic repository analyses."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable

ANALYSIS_CACHE_SCHEMA_VERSION = "entroly.repository-analysis-cache.v1"
_MAX_ANALYSIS_BYTES = 64 * 1024 * 1024
_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class PersistentAnalysisCache:
    """Fail-open immutable cache with an independent envelope commitment."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve() / "analysis"

    @staticmethod
    def _identity(identity: dict[str, object]) -> tuple[str, bytes]:
        canonical = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest(), canonical

    def _path(self, namespace: str, key: str) -> Path:
        if not _NAMESPACE.fullmatch(namespace):
            raise ValueError("invalid analysis cache namespace")
        return self.directory / namespace / key[:2] / f"{key}.json"

    def load(
        self,
        namespace: str,
        identity: dict[str, object],
        *,
        verify: Callable[[dict[str, object]], bool],
    ) -> tuple[dict[str, object] | None, str]:
        key, _ = self._identity(identity)
        target = self._path(namespace, key)
        try:
            if not target.is_file():
                return None, "miss"
            if target.stat().st_size > _MAX_ANALYSIS_BYTES:
                return None, "corrupt"
            envelope = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                return None, "corrupt"
            expected = envelope.pop("cache_sha256", None)
            canonical = json.dumps(
                envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            if (
                envelope.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION
                or envelope.get("namespace") != namespace
                or envelope.get("identity") != identity
                or not isinstance(expected, str)
                or hashlib.sha256(canonical).hexdigest() != expected
            ):
                return None, "corrupt"
            payload = envelope.get("payload")
            if not isinstance(payload, dict) or not verify(payload):
                return None, "corrupt"
            return payload, "hit"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None, "corrupt"

    def store(
        self,
        namespace: str,
        identity: dict[str, object],
        payload: dict[str, object],
        *,
        replace: bool = False,
    ) -> bool:
        key, _ = self._identity(identity)
        target = self._path(namespace, key)
        try:
            envelope: dict[str, object] = {
                "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
                "namespace": namespace,
                "identity": identity,
                "payload": payload,
            }
            canonical = json.dumps(
                envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            envelope["cache_sha256"] = hashlib.sha256(canonical).hexdigest()
            rendered = json.dumps(
                envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
            if len(rendered.encode("utf-8")) > _MAX_ANALYSIS_BYTES:
                return False
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not replace:
                return False
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(rendered)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temporary_name).replace(target)
                return True
            finally:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()
        except OSError:
            return False


__all__ = ["ANALYSIS_CACHE_SCHEMA_VERSION", "PersistentAnalysisCache"]
