"""Build the MCP desktop bundle from its reviewed source manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from _release_artifacts import rebuild_mcpb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="repository root (default: this script's checkout)",
    )
    args = parser.parse_args()
    print(rebuild_mcpb(args.root.resolve()))


if __name__ == "__main__":
    main()
