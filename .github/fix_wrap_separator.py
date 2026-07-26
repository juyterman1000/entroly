from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one block, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "entroly/cli.py",
    '''def _resolved_wrap_env(spec: dict, port: int) -> dict[str, str]:
    """Resolve the complete explicit environment contract for a CLI wrapper."""
    values = {spec["env_key"]: spec["env_val"].format(port=port)}
    for key, value in spec.get("extra_env", {}).items():
        values[str(key)] = str(value).format(port=port)
    return values


def _start_proxy_if_needed(port: int) -> bool:''',
    '''def _resolved_wrap_env(spec: dict, port: int) -> dict[str, str]:
    """Resolve the complete explicit environment contract for a CLI wrapper."""
    values = {spec["env_key"]: spec["env_val"].format(port=port)}
    for key, value in spec.get("extra_env", {}).items():
        values[str(key)] = str(value).format(port=port)
    return values


def _split_wrap_agent_args(agent_args: list[str] | None) -> tuple[list[str], bool]:
    """Consume Entroly's explicit ``--`` separator without forwarding it.

    ``argparse.REMAINDER`` preserves the separator in ``agent_args``. Passing
    that token to clients such as Copilot changes their parsing semantics and
    turns later options into prompt text. When the separator is present,
    everything after it belongs to the client and Entroly must not recover its
    own ``--port``, ``--dry-run``, or ``--force`` flags from that tail.
    """
    values = list(agent_args or ())
    if values and values[0] == "--":
        return values[1:], True
    return values, False


def _start_proxy_if_needed(port: int) -> bool:''',
)

replace_once(
    "entroly/cli.py",
    '''    port = args.port
    # Recover --port if argparse.REMAINDER swallowed it after the agent name.
    if port is None and "--port" in args.agent_args:
        idx = args.agent_args.index("--port")
        if idx + 1 < len(args.agent_args):
            try:
                port = int(args.agent_args[idx + 1])
                args.agent_args.pop(idx)
                args.agent_args.pop(idx)
            except ValueError:
                pass
    port = port or 9377

    # --dry-run can be swallowed by agent_args (argparse.REMAINDER), exactly
    # like --port above. Recover it from either source.
    dry_run = bool(getattr(args, "dry_run", False))
    if "--dry-run" in args.agent_args:
        dry_run = True
        args.agent_args.remove("--dry-run")''',
    '''    agent_args, explicit_separator = _split_wrap_agent_args(args.agent_args)

    port = args.port
    # Without an explicit separator, argparse.REMAINDER may swallow Entroly's
    # own --port after the agent name. With `--`, the tail belongs to the client.
    if not explicit_separator and port is None and "--port" in agent_args:
        idx = agent_args.index("--port")
        if idx + 1 < len(agent_args):
            try:
                port = int(agent_args[idx + 1])
                agent_args.pop(idx)
                agent_args.pop(idx)
            except ValueError:
                pass
    port = port or 9377

    # Recover Entroly's --dry-run only when no explicit client separator exists.
    dry_run = bool(getattr(args, "dry_run", False))
    if not explicit_separator and "--dry-run" in agent_args:
        dry_run = True
        agent_args.remove("--dry-run")''',
)

replace_once(
    "entroly/cli.py",
    '''    force_flag = bool(getattr(args, "force", False))
    if "--force" in args.agent_args:
        force_flag = True
        args.agent_args.remove("--force")''',
    '''    force_flag = bool(getattr(args, "force", False))
    if not explicit_separator and "--force" in agent_args:
        force_flag = True
        agent_args.remove("--force")''',
)

replace_once(
    "entroly/cli.py",
    '        launch = " ".join(spec["cmd"] + (args.agent_args or []))',
    '        launch = " ".join(spec["cmd"] + agent_args)',
)

replace_once(
    "entroly/cli.py",
    '        agent_cmd = spec["cmd"] + args.agent_args',
    '        agent_cmd = spec["cmd"] + agent_args',
)

replace_once(
    "tests/test_agent_compatibility.py",
    'from entroly.cli import _WRAP_AGENTS, _resolved_wrap_env\n',
    'from entroly.cli import _WRAP_AGENTS, _resolved_wrap_env, _split_wrap_agent_args\n',
)

path = Path("tests/test_agent_compatibility.py")
text = path.read_text(encoding="utf-8")
text += '''


def test_wrap_consumes_explicit_separator_and_preserves_client_options():
    client_args, explicit = _split_wrap_agent_args(
        ["--", "--port", "7777", "--force", "--dry-run", "--prompt=Hello world"]
    )
    assert explicit is True
    assert client_args == [
        "--port",
        "7777",
        "--force",
        "--dry-run",
        "--prompt=Hello world",
    ]


def test_wrap_without_separator_keeps_recoverable_entroly_options():
    agent_args, explicit = _split_wrap_agent_args(["--port", "9379", "-p", "hello"])
    assert explicit is False
    assert agent_args == ["--port", "9379", "-p", "hello"]
'''
path.write_text(text, encoding="utf-8")
