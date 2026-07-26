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


def _split_wrap_agent_args(
    agent_args: list[str] | None,
) -> tuple[list[str], list[str], bool]:
    """Split Entroly options from client arguments at the first ``--``.

    ``argparse.REMAINDER`` can return Entroly options followed by the separator,
    for example ``["--port", "9377", "--", "-p", "hello"]``. Entroly may
    recover known wrapper options only from the left side. The separator itself
    is consumed, and the right side is forwarded byte-for-byte as argv tokens.

    When there is no separator, the same mutable list is returned for wrapper
    parsing and client launch so recovered Entroly flags are removed before the
    remaining tokens are forwarded.
    """
    values = list(agent_args or ())
    if "--" in values:
        index = values.index("--")
        return values[:index], values[index + 1 :], True
    return values, values, False


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
    '''    wrapper_args, agent_args, explicit_separator = _split_wrap_agent_args(
        args.agent_args
    )

    port = args.port
    # argparse.REMAINDER may swallow Entroly's own --port after the agent name.
    # Only the left side of an explicit separator belongs to Entroly.
    if port is None and "--port" in wrapper_args:
        idx = wrapper_args.index("--port")
        if idx + 1 < len(wrapper_args):
            try:
                port = int(wrapper_args[idx + 1])
                wrapper_args.pop(idx)
                wrapper_args.pop(idx)
            except ValueError:
                pass
    port = port or 9377

    # Recover Entroly's --dry-run only from the wrapper side of the separator.
    dry_run = bool(getattr(args, "dry_run", False))
    if "--dry-run" in wrapper_args:
        dry_run = True
        wrapper_args.remove("--dry-run")''',
)

replace_once(
    "entroly/cli.py",
    '''    force_flag = bool(getattr(args, "force", False))
    if "--force" in args.agent_args:
        force_flag = True
        args.agent_args.remove("--force")''',
    '''    force_flag = bool(getattr(args, "force", False))
    if "--force" in wrapper_args:
        force_flag = True
        wrapper_args.remove("--force")''',
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


def test_wrap_splits_wrapper_options_from_client_arguments():
    wrapper_args, client_args, explicit = _split_wrap_agent_args(
        [
            "--port",
            "7777",
            "--force",
            "--",
            "--port",
            "8888",
            "--dry-run",
            "--prompt=Hello world",
        ]
    )
    assert explicit is True
    assert wrapper_args == ["--port", "7777", "--force"]
    assert client_args == [
        "--port",
        "8888",
        "--dry-run",
        "--prompt=Hello world",
    ]


def test_wrap_consumes_separator_when_it_is_first():
    wrapper_args, client_args, explicit = _split_wrap_agent_args(
        ["--", "-s", "--prompt=Hello world"]
    )
    assert explicit is True
    assert wrapper_args == []
    assert client_args == ["-s", "--prompt=Hello world"]


def test_wrap_without_separator_shares_recoverable_argument_list():
    wrapper_args, client_args, explicit = _split_wrap_agent_args(
        ["--port", "9379", "-p", "hello"]
    )
    assert explicit is False
    assert wrapper_args is client_args
    wrapper_args.pop(0)
    wrapper_args.pop(0)
    assert client_args == ["-p", "hello"]
'''
path.write_text(text, encoding="utf-8")
