"""Container PID 1 dispatcher with explicit, validated service modes."""

from __future__ import annotations

import os
import sys


def _normalized_args(argv: list[str]) -> list[str]:
    args = list(argv)
    if args and args[0] == "serve":
        args.pop(0)
    return args


def main(argv: list[str] | None = None) -> None:
    args = _normalized_args(list(sys.argv[1:] if argv is None else argv))
    proxy_mode = "--proxy" in args or os.environ.get("ENTROLY_PROXY") == "1"
    args = [item for item in args if item != "--proxy"]

    if proxy_mode:
        if args:
            raise SystemExit(
                "container proxy mode accepts configuration through ENTROLY_* "
                f"environment variables; unsupported arguments: {args!r}"
            )
        os.environ.setdefault("ENTROLY_CONTAINER_MODE", "proxy")
        # Safe default for Linux --network=host. Bridge users must deliberately
        # select a remote bind and satisfy the authenticated remote-proxy contract.
        os.environ.setdefault("ENTROLY_PROXY_HOST", "127.0.0.1")
        from .container_proxy import main as proxy_main

        proxy_main()
        return

    allowed = {"--sse"}
    unknown = [item for item in args if item not in allowed]
    if unknown:
        raise SystemExit(f"unsupported Entroly container arguments: {unknown!r}")
    os.environ.setdefault("ENTROLY_CONTAINER_MODE", "mcp")
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *args]
        from .server import main as server_main

        server_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()


__all__ = ["main"]
