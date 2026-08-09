from __future__ import annotations

from pathlib import Path

import entroly.repository_intelligence as repo
from entroly.repository_intelligence.models import FileRecord, RepositoryLimits
from entroly.repository_intelligence.parsers import ParsedFile


def _parsed(path: str, language: str, imports: set[str] | None = None) -> ParsedFile:
    values = imports or set()
    return ParsedFile(
        record=FileRecord(
            path=path,
            language=language,
            sha256="0" * 64,
            byte_length=1,
            line_count=1,
            is_test=False,
            imports=tuple(sorted(values)),
        ),
        symbols=[],
        imports=set(values),
        import_aliases={},
        calls=[],
    )


def test_repository_index_unions_legacy_and_universal_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parsed = {
        "src/main.zig": _parsed("src/main.zig", "zig", {"util.zig"}),
        "src/util.zig": _parsed("src/util.zig", "zig"),
        "legacy.py": _parsed("legacy.py", "python"),
    }
    monkeypatch.setattr(
        repo,
        "scan_repository",
        lambda root, policy: (parsed, []),
    )
    monkeypatch.setattr(
        repo,
        "resolve_imports",
        lambda files: {
            "src/main.zig": (),
            "src/util.zig": (),
            "legacy.py": ("src/util.zig",),
        },
    )
    monkeypatch.setattr(repo, "resolve_calls", lambda files, symbols, policy: ((), ()))

    index = repo.build_repository_index(tmp_path, limits=RepositoryLimits())

    # New language-neutral edge is added.
    assert index.file_dependencies["src/main.zig"] == ("src/util.zig",)
    # Existing resolver evidence is never removed by universalization.
    assert index.file_dependencies["legacy.py"] == ("src/util.zig",)

    impact = repo.analyze_change_impact(index, ["src/util.zig"])
    assert set(impact.impacted_paths) >= {
        "src/util.zig",
        "src/main.zig",
        "legacy.py",
    }
