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
    assert scripts["entroly-compression-mcp"] == "entroly.compression_mcp:main"


def test_root_pyproject_defines_documented_full_extra() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    full_section = text.split("full = [", 1)[1].split("]", 1)[0]

    for dependency in ("cryptography", "entroly-core", "httpx", "starlette", "uvicorn"):
        assert dependency in full_section


def test_root_pyproject_keeps_native_engine_optional() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    hard_deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    native_extra = text.split("native = [", 1)[1].split("]", 1)[0]

    assert "mcp" in hard_deps
    assert "entroly-core" not in hard_deps
    assert "entroly-core" in native_extra


def test_root_pyproject_defines_test_extra() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    test_section = text.split("test = [", 1)[1].split("]", 1)[0]

    assert "pytest" in test_section


def test_nested_pyproject_dependency_shape_matches_root() -> None:
    root = Path("pyproject.toml").read_text(encoding="utf-8")
    nested = Path("entroly/pyproject.toml").read_text(encoding="utf-8")

    for text in (root, nested):
        hard_deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
        native_extra = text.split("native = [", 1)[1].split("]", 1)[0]
        full_extra = text.split("full = [", 1)[1].split("]", 1)[0]
        test_extra = text.split("test = [", 1)[1].split("]", 1)[0]

        assert "mcp" in hard_deps
        assert "entroly-core" not in hard_deps
        assert "entroly-core" in native_extra
        assert "entroly-core" in full_extra
        assert "pytest" in test_extra
