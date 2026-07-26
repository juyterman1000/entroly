from pathlib import Path

cli_path = Path("entroly/cli.py")
cli_text = cli_path.read_text(encoding="utf-8")
old_binding = "wrapper_args, agent_args, explicit_separator = _split_wrap_agent_args(\n"
new_binding = "wrapper_args, agent_args, _explicit_separator = _split_wrap_agent_args(\n"
if cli_text.count(old_binding) != 1:
    raise RuntimeError(
        f"expected one wrapper separator binding, found {cli_text.count(old_binding)}"
    )
cli_path.write_text(cli_text.replace(old_binding, new_binding, 1), encoding="utf-8")

readme_path = Path("README.md")
readme = readme_path.read_text(encoding="utf-8")
old_boundary = (
    "GitHub-hosted subscription inference is not claimed as proxied. |"
)
new_boundary = (
    "Entroly does not claim interception of GitHub-hosted subscription inference. |"
)
if readme.count(old_boundary) != 1:
    raise RuntimeError(
        f"expected one Copilot subscription boundary, found {readme.count(old_boundary)}"
    )
readme_path.write_text(
    readme.replace(old_boundary, new_boundary, 1),
    encoding="utf-8",
)
