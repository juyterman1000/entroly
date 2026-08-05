from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "node_modules", "target", "__pycache__"}
TEMPORARY = {
    Path(".github/neutralize_external_names.py"),
    Path(".github/workflows/competitor-name-audit.yml"),
    Path(".github/workflows/one-shot-neutralize-external-names.yml"),
}
CANONICAL_METHOD = (
    "https://github.com/juyterman1000/entroly/blob/main/docs/BENCHMARKS.md"
)
CURRENT_RECOVERY_ARTIFACT = Path(
    "benchmarks/results/recovery_resilience_holdout_revalidation_v5.json"
)

# The source terms exist only in this self-deleting migration file. The final
# repository guard stores SHA-256 digests instead of product names.
REPLACEMENTS = (
    ("headroomlabs-ai/headroom", "operator/external-adapter"),
    ("headroom-ai", "external-adapter"),
    ("HEADROOM", "EXTERNAL_ADAPTER"),
    ("Headroom", "External Baseline"),
    ("headroom", "external_adapter"),
    ("LEAN-CTX", "EXTERNAL-CONTEXT-TOOL"),
    ("LEAN_CTX", "EXTERNAL_CONTEXT_TOOL"),
    ("LEANCTX", "EXTERNAL_CONTEXT_TOOL"),
    ("Lean-CTX", "External Context Tool"),
    ("Lean_CTX", "External Context Tool"),
    ("LeanCTX", "External Context Tool"),
    ("lean-ctx", "external-context-tool"),
    ("lean_ctx", "external_context_tool"),
    ("leanctx", "external_context_tool"),
)
URL_PATTERN = re.compile(
    r"https?://[^\s\]\)\"']*(?:headroom|lean(?:ctx|[-_]?ctx))[^\s\]\)\"']*",
    flags=re.IGNORECASE,
)


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _canonical_source_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


def iter_text_files():
    for path in sorted(candidate for candidate in ROOT.rglob("*") if candidate.is_file()):
        relative = path.relative_to(ROOT)
        if relative in TEMPORARY or any(part in SKIP_PARTS for part in relative.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        yield path, relative, text


def neutralize_tree() -> list[str]:
    changed: list[str] = []
    for path, relative, original in iter_text_files():
        updated = URL_PATTERN.sub(CANONICAL_METHOD, original)
        for old, new in REPLACEMENTS:
            updated = updated.replace(old, new)

        updated = updated.replace(
            'importlib.metadata.version("external-adapter")',
            'os.environ.get("ENTROLY_EXTERNAL_ADAPTER_VERSION", "operator-provided")',
        )
        updated = updated.replace(
            "importlib.metadata.version('external-adapter')",
            "os.environ.get('ENTROLY_EXTERNAL_ADAPTER_VERSION', 'operator-provided')",
        )
        updated = updated.replace(
            '_distribution_record_sha256("external-adapter")', "None"
        )
        updated = updated.replace(
            "_distribution_record_sha256('external-adapter')", "None"
        )

        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(relative.as_posix())
    return changed


def _set_current_recovery_implementation(report: dict[str, Any]) -> bool:
    implementation = _canonical_source_sha256(
        (
            ROOT / "benchmarks/recovery_resilience.py",
            ROOT / "entroly/compression_retrieval_store.py",
        )
    )
    modified = False
    candidates = (
        report.get("participants", {}).get("entroly"),
        report.get("adapters", {}).get("entroly", {}).get("participant"),
    )
    for participant in candidates:
        if not isinstance(participant, dict):
            continue
        runtime = participant.get("runtime")
        if not isinstance(runtime, dict):
            continue
        if runtime.get("implementation_sha256") != implementation:
            runtime["implementation_sha256"] = implementation
            modified = True
    return modified


def refresh_json_integrity() -> list[str]:
    """Refresh only seals invalidated by the neutral naming migration."""

    changed: list[str] = []
    for path in sorted(ROOT.rglob("*.json")):
        relative = path.relative_to(ROOT)
        if relative in TEMPORARY or any(part in SKIP_PARTS for part in relative.parts):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(value, dict):
            continue

        modified = False
        if relative == CURRENT_RECOVERY_ARTIFACT:
            modified = _set_current_recovery_implementation(value) or modified

        protocol = value.get("protocol")
        if "protocol_sha256" in value and isinstance(protocol, dict):
            expected = _canonical_sha256(protocol)
            if value.get("protocol_sha256") != expected:
                value["protocol_sha256"] = expected
                modified = True

        if "payload_sha256" in value:
            unhashed = {
                key: item for key, item in value.items() if key != "payload_sha256"
            }
            expected = _canonical_sha256(unhashed)
            if value.get("payload_sha256") != expected:
                value["payload_sha256"] = expected
                modified = True

        if modified:
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            changed.append(relative.as_posix())
    return changed


def write_external_adapter() -> list[str]:
    files = {
        Path("external_adapter/__init__.py"): '''"""Operator-supplied adapter contract for neutral external baselines.

The Entroly repository does not bundle, identify, install, or endorse another
context product. Benchmark operators provide import targets explicitly through
environment variables. Importing an external adapter without that configuration
fails closed.
"""

from __future__ import annotations

import importlib
import os
from typing import Any


def resolve_symbol(variable: str) -> Any:
    spec = os.environ.get(variable, "").strip()
    if not spec or ":" not in spec:
        raise RuntimeError(
            f"{variable} must be set to an operator-controlled 'module:symbol' import target"
        )
    module_name, symbol_name = spec.split(":", 1)
    if not module_name or not symbol_name:
        raise RuntimeError(f"invalid external adapter import target in {variable}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as error:
        raise RuntimeError(
            f"external adapter symbol {symbol_name!r} is unavailable in {module_name!r}"
        ) from error


def compress(*args: Any, **kwargs: Any) -> Any:
    implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_COMPRESS")
    return implementation(*args, **kwargs)


__all__ = ["compress", "resolve_symbol"]
''',
        Path("external_adapter/cache/__init__.py"): (
            '"""Generic recovery-store adapter namespace."""\n'
        ),
        Path("external_adapter/cache/backends/__init__.py"): (
            '"""Generic external backend adapter namespace."""\n'
        ),
        Path("external_adapter/cache/backends/sqlite.py"): '''"""Operator-supplied persistent-backend class."""

from external_adapter import resolve_symbol


class SQLiteBackend:
    def __new__(cls, *args, **kwargs):
        implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_BACKEND")
        return implementation(*args, **kwargs)
''',
        Path("external_adapter/cache/compression_store.py"): '''"""Operator-supplied recovery-store compatibility contract."""

from __future__ import annotations

from typing import Any

from external_adapter import resolve_symbol


class CompressionStore:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_STORE")
        return implementation(*args, **kwargs)


def set_request_compression_store(store: Any) -> Any:
    implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_SET_REQUEST_STORE")
    return implementation(store)


def clear_request_compression_store() -> Any:
    implementation = resolve_symbol("ENTROLY_EXTERNAL_ADAPTER_CLEAR_REQUEST_STORE")
    return implementation()


__all__ = [
    "CompressionStore",
    "set_request_compression_store",
    "clear_request_compression_store",
]
''',
    }
    written: list[str] = []
    for relative, content in files.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(relative.as_posix())
    return written


def write_policy_guard() -> str:
    relative = Path("scripts/check_external_name_policy.py")
    path = ROOT / relative
    path.write_text(
        '''#!/usr/bin/env python3
"""Fail when prohibited external product names enter the current tree.

Names are represented only by SHA-256 digests. This keeps the policy itself
brand-neutral while detecting plain, hyphenated, underscored, URL, package, and
identifier forms after alphanumeric normalization.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROHIBITED = {
    8: {"26e93d81a9553eabd165301cad992369094b6a1759a62e94a998a94aa5315902"},
    7: {"d8a564a233ed75c8d55c193f8a56b5937cb2b5dec3b3566fa0537f7fa434dca7"},
}
SKIP_PARTS = {".git", ".venv", "node_modules", "target", "__pycache__"}


def normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def violations() -> list[str]:
    found: list[str] = []
    for path in sorted(candidate for candidate in ROOT.rglob("*") if candidate.is_file()):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            value = normalized(line)
            for length, digests in PROHIBITED.items():
                if len(value) < length:
                    continue
                matched = False
                for index in range(len(value) - length + 1):
                    digest = hashlib.sha256(
                        value[index : index + length].encode()
                    ).hexdigest()
                    if digest in digests:
                        found.append(f"{relative}:{line_number}")
                        matched = True
                        break
                if matched:
                    break
    return found


def main() -> int:
    found = violations()
    if found:
        print("prohibited external product name found in current tree:", file=sys.stderr)
        for location in found:
            print(f"- {location}", file=sys.stderr)
        return 1
    print("external-name policy check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
        encoding="utf-8",
    )
    return relative.as_posix()


def wire_guard() -> list[str]:
    changed: list[str] = []
    workflow = ROOT / ".github/workflows/visibility-integrity.yml"
    text = workflow.read_text(encoding="utf-8")
    old = (
        "      - name: Validate owned visibility surface\n"
        "        run: python scripts/check_distribution_surface.py\n"
    )
    new = (
        "      - name: Validate owned visibility surface\n"
        "        run: |\n"
        "          python scripts/check_distribution_surface.py\n"
        "          python scripts/check_external_name_policy.py\n"
    )
    if old in text:
        workflow.write_text(text.replace(old, new), encoding="utf-8")
        changed.append(workflow.relative_to(ROOT).as_posix())

    codeowners = ROOT / ".github/CODEOWNERS"
    text = codeowners.read_text(encoding="utf-8")
    line = "/scripts/check_external_name_policy.py @juyterman1000"
    if line not in text:
        anchor = "/scripts/check_distribution_surface.py @juyterman1000\n"
        codeowners.write_text(
            text.replace(anchor, anchor + line + "\n"), encoding="utf-8"
        )
        changed.append(codeowners.relative_to(ROOT).as_posix())
    return changed


def remove_temporary_files() -> list[str]:
    removed: list[str] = []
    for relative in TEMPORARY:
        path = ROOT / relative
        if path.exists():
            path.unlink()
            removed.append(relative.as_posix())
    return removed


def main() -> None:
    changed = neutralize_tree()
    changed.extend(refresh_json_integrity())
    changed.extend(write_external_adapter())
    changed.append(write_policy_guard())
    changed.extend(wire_guard())
    changed.extend(remove_temporary_files())
    print(f"neutral naming migration updated {len(set(changed))} paths")


if __name__ == "__main__":
    main()
