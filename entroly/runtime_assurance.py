"""Install Entroly assurance sidecars for CLI-hosted proxy surfaces."""

from __future__ import annotations

from collections.abc import Sequence


def install_runtime_assurance(argv: Sequence[str]) -> None:
    """Install one authoritative assurance stack for the requested surface."""
    command = str(argv[0]) if argv else ""

    # Wrap itself does not handle provider traffic. Only serialize startup; its
    # child enters through proxy_cli_entry and installs the proxy assurance stack.
    if command == "wrap":
        from .cli_startup_serialization import install_cli_startup_serialization

        install_cli_startup_serialization()
        return

    # Explicit routing uses container_proxy, whose import order also includes
    # transport/control/access-security wrappers. Do not pre-wrap that stack.
    if command == "proxy" and "--routing" in argv:
        return

    if command not in {"proxy", "go"}:
        return

    # Recovery must be installed before proxy.py binds the store class.
    from . import compression_retrieval_store_resilient as _recovery  # noqa: F401
    from . import proxy_generation_routes as _routes  # noqa: F401
    from .anthropic_state_contract import install_active_tool_state_proof
    from .semantic_assurance import install_proxy_semantic_assurance

    install_active_tool_state_proof()
    install_proxy_semantic_assurance()


__all__ = ["install_runtime_assurance"]
