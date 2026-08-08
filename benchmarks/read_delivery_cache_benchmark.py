#!/usr/bin/env python3
"""Wire-level cold/warm file-read benchmark with failure-closed accounting.

Both systems read byte-identical, run-unique files. Entroly is charged for its
complete agent-visible tool response. External commands must exit successfully;
an empty/error response is never scored as perfect compression.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tiktoken


REPO = Path(__file__).resolve().parent.parent
ENCODING = tiktoken.get_encoding("o200k_base")


def _tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _external_read(binary: str, cwd: Path, relative_path: str) -> str:
    proc = subprocess.run(
        [binary, "read", relative_path, "-m", "full"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"external read failed ({proc.returncode}): {proc.stderr[-500:]}"
        )
    if not proc.stdout:
        raise RuntimeError("external read returned empty stdout")
    return proc.stdout


def _pick_files(limit: int) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "entroly/*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    picked: list[Path] = []
    for relative in listed:
        path = REPO / relative
        if path.is_file() and 6_000 <= path.stat().st_size <= 60_000:
            picked.append(path)
        if len(picked) >= limit:
            break
    return picked


def run(limit: int, binary: str | None) -> dict[str, Any]:
    temp_parent = REPO / ".tmp"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="read-cache-", dir=temp_parent) as raw_root:
        root = Path(raw_root)
        nonce = uuid.uuid4().hex
        files: list[Path] = []
        for index, source_path in enumerate(_pick_files(limit)):
            target = root / f"sample_{index}.py"
            text = source_path.read_text(encoding="utf-8", errors="replace")
            target.write_text(
                f"# cold-run {nonce}-{index}\n{text}",
                encoding="utf-8",
                newline="",
            )
            files.append(target)

        os.environ["ENTROLY_SOURCE"] = str(root)
        os.environ["ENTROLY_DIR"] = str(root / "state")
        from entroly.server import create_mcp_server

        mcp, _ = create_mcp_server(allowed_tools={"smart_read"})
        smart_read = mcp._tool_manager._tools["smart_read"].fn
        ctx = SimpleNamespace(session=object(), client_id="benchmark")

        entroly_cold = entroly_warm = 0
        external_cold = external_warm = 0
        raw_tokens = 0
        rows: list[dict[str, Any]] = []

        for path in files:
            relative = path.name
            source = path.read_text(encoding="utf-8", errors="replace")
            raw = _tokens(source)
            raw_tokens += raw

            cold_response = smart_read(str(path), ctx, resolution="full")
            warm_response = smart_read(str(path), ctx, resolution="full")
            if cold_response.startswith("~") or not warm_response.startswith("~"):
                raise RuntimeError("Entroly cold/warm cache contract was not met")
            ent_cold = _tokens(cold_response)
            ent_warm = _tokens(warm_response)
            entroly_cold += ent_cold
            entroly_warm += ent_warm

            ext_cold = ext_warm = None
            if binary:
                ext_cold_text = _external_read(binary, root, relative)
                ext_warm_text = _external_read(binary, root, relative)
                ext_cold = _tokens(ext_cold_text)
                ext_warm = _tokens(ext_warm_text)
                external_cold += ext_cold
                external_warm += ext_warm

            rows.append({
                "sample": relative,
                "raw_tokens": raw,
                "entroly_cold_wire_tokens": ent_cold,
                "entroly_warm_wire_tokens": ent_warm,
                "external_cold_wire_tokens": ext_cold,
                "external_warm_wire_tokens": ext_warm,
            })

        return {
            "tokenizer": "o200k_base",
            "files": len(files),
            "raw_tokens": raw_tokens,
            "entroly": {
                "cold_wire_tokens": entroly_cold,
                "warm_wire_tokens": entroly_warm,
                "warm_reduction_vs_raw": (
                    1 - entroly_warm / raw_tokens if raw_tokens else 0.0
                ),
            },
            "external": ({
                "binary": binary,
                "version": subprocess.run(
                    [binary, "--version"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                ).stdout.strip(),
                "cold_wire_tokens": external_cold,
                "warm_wire_tokens": external_warm,
                "warm_reduction_vs_raw": (
                    1 - external_warm / raw_tokens if raw_tokens else 0.0
                ),
            } if binary else None),
            "rows": rows,
            "caveat": (
                "Measures agent-visible wire text, not provider billing or answer quality."
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--external-bin", default=os.environ.get("ENTROLY_EXTERNAL_CTX_BIN"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    binary = args.external_bin
    if binary and not Path(binary).is_file():
        resolved = shutil.which(binary)
        if not resolved:
            raise SystemExit(f"external binary not found: {binary}")
        binary = resolved

    payload = run(args.limit, binary)
    rendered = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
