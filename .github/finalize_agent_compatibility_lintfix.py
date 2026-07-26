from pathlib import Path

path = Path("entroly/cli.py")
text = path.read_text(encoding="utf-8")
old = "wrapper_args, agent_args, explicit_separator = _split_wrap_agent_args(\n"
new = "wrapper_args, agent_args, _explicit_separator = _split_wrap_agent_args(\n"
if text.count(old) != 1:
    raise RuntimeError(f"expected one wrapper separator binding, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
