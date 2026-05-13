"""Console entry point with a safe native-binary escape hatch.

The default path intentionally runs the Python CLI. A future wheel may bundle a
native ``entroly-rs`` binary; when explicitly enabled, this module can exec it
without changing the public ``entroly`` command.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _candidate_binaries() -> list[Path]:
    exe = "entroly-rs.exe" if os.name == "nt" else "entroly-rs"
    env_bin = os.environ.get("ENTROLY_RUST_BIN")
    candidates: list[Path] = []
    if env_bin:
        candidates.append(Path(env_bin))
    candidates.append(Path(__file__).resolve().parent / "bin" / exe)
    return candidates


def _run_python_cli() -> int:
    from entroly.cli import main

    result = main()
    return int(result or 0)


def main() -> None:
    """Run the bundled/native launcher when requested, otherwise Python CLI."""
    use_native = os.environ.get("ENTROLY_USE_RUST_LAUNCHER", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_native:
        for candidate in _candidate_binaries():
            if candidate.exists():
                result = subprocess.run([str(candidate), *sys.argv[1:]], check=False)
                raise SystemExit(result.returncode)
        print(
            "[entroly] ENTROLY_USE_RUST_LAUNCHER=1 was set, but no entroly-rs binary was found.",
            file=sys.stderr,
        )
        raise SystemExit(127)

    raise SystemExit(_run_python_cli())


if __name__ == "__main__":
    main()
