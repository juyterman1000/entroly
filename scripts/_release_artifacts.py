"""Build derived release artifacts from their reviewed source manifests."""

from __future__ import annotations

import zipfile
from pathlib import Path


MCPB_MANIFEST = Path(".mcpb-build/manifest.json")
MCPB_BUNDLE = Path("entroly.mcpb")


def rebuild_mcpb(root: Path) -> Path:
    """Atomically rebuild the MCP bundle with reproducible ZIP metadata."""
    source = root / MCPB_MANIFEST
    target = root / MCPB_BUNDLE
    temporary = target.with_name(f".{target.name}.tmp")

    if not source.is_file():
        raise FileNotFoundError(f"MCP bundle manifest is missing: {source}")

    info = zipfile.ZipInfo("manifest.json", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16

    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr(info, source.read_bytes())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    return target
