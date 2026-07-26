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
    '''        "extra_env": {"COPILOT_PROVIDER_TYPE": "openai"},
        "api_key_env": "COPILOT_PROVIDER_API_KEY",
        "subscription_alt": "entroly attach create --client copilot --project . --ttl 4h --install",''',
    '''        "extra_env": {"COPILOT_PROVIDER_TYPE": "openai"},
        "subscription_alt": "entroly attach create --client copilot --project . --ttl 4h --install",''',
)

replace_once(
    "tests/test_agent_compatibility.py",
    '    assert spec["api_key_env"] == "COPILOT_PROVIDER_API_KEY"\n',
    '    assert "api_key_env" not in spec  # local providers may be unauthenticated\n',
)
