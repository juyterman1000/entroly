"""Fail-closed runtime fingerprints for bounded benchmark continuations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any


_ORACLE_IMAGE = "entroly-dogfood-real-oracle:py312"
_MAX_VERSION_BYTES = 64 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_ollama_opener(base_url: str):
    """Validate a loopback endpoint and return a transport without proxies or redirects."""
    try:
        parsed = urllib.parse.urlsplit(base_url)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.port is not None
            and parsed.port > 0
            and parsed.username is None
            and parsed.password is None
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
            and not any(c.isspace() or c == "\\" for c in base_url)
            and "?" not in base_url
            and "#" not in base_url
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("runtime identity requires a local Ollama HTTP endpoint")

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _ollama_version(base_url: str) -> str:
    opener = local_ollama_opener(base_url)
    url = f"{base_url.rstrip('/')}/api/version"
    try:
        with opener.open(url, timeout=10) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("Ollama version response must not redirect")
            body = response.read(_MAX_VERSION_BYTES + 1)
        if len(body) > _MAX_VERSION_BYTES:
            raise ValueError("Ollama version response exceeds the size limit")
        payload = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise RuntimeError("unable to read local Ollama version") from None
    version = payload.get("version") if isinstance(payload, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Ollama version must be a nonempty string")
    return version


def _native_files(engine: Any) -> dict[str, str]:
    if not getattr(engine, "path", None):
        raise ValueError("loaded native extension path is missing")
    try:
        loaded = Path(engine.path).resolve(strict=True)
        root = loaded.parent
        init = root / "__init__.py"
        is_binary = any(loaded.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
        if loaded.name != "__init__.py" and not is_binary:
            raise ValueError("loaded native extension path is not a binary or package")
        if loaded.name == "__init__.py" or init.is_file():
            binaries = [
                path for path in root.rglob("*")
                if path.is_file()
                and any(path.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)
            ]
            files = [init, *binaries]
        else:
            binaries = [loaded]
            files = binaries
        if not binaries:
            raise ValueError("loaded native package has no extension binary")
        hashes = {}
        for path in sorted(files, key=lambda path: path.relative_to(root).as_posix()):
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("native extension file escapes its module package")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
        return hashes
    except OSError:
        raise RuntimeError("unable to fingerprint loaded native extension files") from None


def runtime_identity(repo: Path, engine: Any, base_url: str) -> dict[str, Any]:
    """Return stable JSON data after the caller validates the local model digest."""
    if os.environ.get("ENTROLY_REAL_COMMIT_SANDBOX") != "1":
        raise ValueError("runtime identity requires ENTROLY_REAL_COMMIT_SANDBOX=1")
    try:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal",
             "--", "benchmarks", "entroly"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("unable to verify clean benchmark and Entroly source") from None
    if status.stdout.strip():
        raise ValueError("modified or untracked source under benchmarks/ or entroly/")
    native = _native_files(engine)
    try:
        image = subprocess.run(
            ["docker", "image", "inspect", _ORACLE_IMAGE, "--format", "{{.Id}}"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("unable to inspect the local oracle Docker image") from None
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", image):
        raise ValueError("oracle Docker image must have an immutable sha256 ID")
    versions = {}
    for name in ("tiktoken", "regex"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # Absence is part of the fingerprint too. Optional tokenizers must
            # not become a new installation requirement for this pilot.
            versions[name] = None
        if versions[name] is not None and not versions[name].strip():
            raise ValueError(f"runtime identity requires a nonempty {name} version")
    return {
        "schema": "entroly.pilot-runtime-identity.v1",
        "oracle_image": image.lower(),
        "ollama_version": _ollama_version(base_url),
        "native": {"files_sha256": native},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "distributions": versions,
    }
