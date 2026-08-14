"""Assurance bootstrap for proxy children spawned by ``entroly wrap``."""

from __future__ import annotations

import sys


def main() -> None:
    # Install recovery before proxy import; route and semantic sidecars then bind
    # to the same CLI proxy implementation users already receive.
    from . import compression_retrieval_store_resilient as _recovery  # noqa: F401
    from . import proxy_generation_routes as _routes  # noqa: F401
    from .anthropic_state_contract import install_active_tool_state_proof
    from .semantic_assurance import install_proxy_semantic_assurance

    install_active_tool_state_proof()
    install_proxy_semantic_assurance()

    from .cli import main as cli_main

    sys.argv = [sys.argv[0], "proxy", *sys.argv[1:]]
    cli_main()


if __name__ == "__main__":
    main()


__all__ = ["main"]
