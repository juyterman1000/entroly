"""Install Entroly assurance sidecars for CLI-hosted proxy surfaces."""

from __future__ import annotations

from collections.abc import Sequence

_PROXY_COMMANDS = frozenset({"proxy", "go", "wrap"})


def install_runtime_assurance(argv: Sequence[str]) -> None:
    """Install only the sidecars needed by the requested CLI surface."""
    command = str(argv[0]) if argv else ""
    if command not in _PROXY_COMMANDS:
        return

    # Recovery must be installed before proxy.py binds the store class.
    from . import compression_retrieval_store_resilient as _recovery  # noqa: F401
    from . import proxy_generation_routes as _routes  # noqa: F401
    from .semantic_assurance import install_proxy_semantic_assurance

    install_proxy_semantic_assurance()

    if command == "wrap":
        from .cli_startup_serialization import install_cli_startup_serialization

        install_cli_startup_serialization()


__all__ = ["install_runtime_assurance"]
