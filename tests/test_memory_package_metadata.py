from __future__ import annotations

from pathlib import Path


def _read_project_scripts(path: str) -> dict[str, str]:
    """Read [project.scripts] entries without requiring Python 3.11 tomllib.

    Python 3.10 does not ship tomllib, and this tiny parser is enough for the
    simple string-to-string console-script metadata we need to guard.
    """
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            continue
        if not in_scripts or "=" not in line:
            continue
        key, value = line.split("=", 1)
        scripts[key.strip()] = value.strip().strip('"')
    return scripts


def test_root_pyproject_exposes_memory_entrypoints() -> None:
    scripts = _read_project_scripts("pyproject.toml")

    assert scripts["entroly"] == "entroly._docker_launcher:launch"
    assert scripts["entroly-memory"] == "entroly.memory_cli:main"


def test_nested_pyproject_exposes_memory_entrypoint() -> None:
    scripts = _read_project_scripts("entroly/pyproject.toml")

    assert scripts["entroly-memory"] == "entroly.memory_cli:main"


def test_root_pyproject_defines_documented_full_extra() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    full_section = text.split("full = [", 1)[1].split("]", 1)[0]

    for dependency in ("cryptography", "httpx", "starlette", "uvicorn"):
        assert dependency in full_section
