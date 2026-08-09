from __future__ import annotations

from entroly.repository_intelligence.models import FileRecord
from entroly.repository_intelligence.parsers import ParsedFile
from entroly.repository_intelligence.workspace_dependencies import (
    DependencyTrust,
    resolve_workspace_dependencies,
)


def _file(path: str, language: str, imports: set[str] | None = None) -> ParsedFile:
    values = imports or set()
    return ParsedFile(
        FileRecord(
            path=path,
            language=language,
            sha256="0" * 64,
            byte_length=1,
            line_count=1,
            is_test=False,
            imports=tuple(sorted(values)),
        ),
        [],
        set(values),
        {},
        [],
    )


def test_zig_file_import_resolves_exactly_relative_to_importer() -> None:
    parsed = {
        "src/main.zig": _file("src/main.zig", "zig", {"util.zig", "std"}),
        "src/util.zig": _file("src/util.zig", "zig"),
    }
    graph = resolve_workspace_dependencies(parsed)
    assert [(edge.importer, edge.target, edge.trust) for edge in graph.dependencies] == [
        ("src/main.zig", "src/util.zig", DependencyTrust.EXACT_RELATIVE)
    ]
    assert all(item.imported_text != "std" for item in graph.unresolved)


def test_c_quoted_header_resolves_local_file_without_inventing_system_paths() -> None:
    parsed = {
        "src/main.c": _file("src/main.c", "c", {"helper.h", "<stdio.h>"}),
        "src/helper.h": _file("src/helper.h", "c"),
    }
    graph = resolve_workspace_dependencies(parsed)
    assert graph.file_dependencies() == {"src/main.c": ("src/helper.h",)}
    assert all(item.imported_text != "<stdio.h>" for item in graph.unresolved)


def test_extensionless_relative_import_is_language_neutral() -> None:
    parsed = {
        "web/main.ts": _file("web/main.ts", "typescript", {"./util"}),
        "web/util.ts": _file("web/util.ts", "typescript"),
    }
    graph = resolve_workspace_dependencies(parsed)
    assert len(graph.dependencies) == 1
    assert graph.dependencies[0].target == "web/util.ts"
    assert graph.dependencies[0].trust is DependencyTrust.UNIQUE_STEM


def test_dotted_module_resolution_preserves_existing_python_behavior() -> None:
    parsed = {
        "app/main.py": _file("app/main.py", "python", {"pkg.service"}),
        "pkg/service.py": _file("pkg/service.py", "python"),
    }
    graph = resolve_workspace_dependencies(parsed)
    assert graph.file_dependencies() == {"app/main.py": ("pkg/service.py",)}
    assert graph.dependencies[0].trust is DependencyTrust.UNIQUE_MODULE


def test_ambiguous_workspace_suffix_is_not_promoted_to_dependency() -> None:
    parsed = {
        "src/main.c": _file("src/main.c", "c", {"common/config.h"}),
        "a/common/config.h": _file("a/common/config.h", "c"),
        "b/common/config.h": _file("b/common/config.h", "c"),
    }
    graph = resolve_workspace_dependencies(parsed)
    assert graph.dependencies == ()
    assert len(graph.unresolved) == 1
    unresolved = graph.unresolved[0]
    assert unresolved.reason == "ambiguous-workspace-suffix"
    assert unresolved.candidates == ("a/common/config.h", "b/common/config.h")


def test_unique_workspace_suffix_is_explicitly_lower_trust_than_relative() -> None:
    parsed = {
        "src/main.c": _file("src/main.c", "c", {"generated/api.h"}),
        "build/generated/api.h": _file("build/generated/api.h", "c"),
    }
    graph = resolve_workspace_dependencies(parsed)
    assert len(graph.dependencies) == 1
    edge = graph.dependencies[0]
    assert edge.target == "build/generated/api.h"
    assert edge.trust is DependencyTrust.UNIQUE_SUFFIX


def test_same_stem_in_multiple_languages_stays_ambiguous() -> None:
    parsed = {
        "src/main.future": _file("src/main.future", "unknown", {"./util"}),
        "src/util.zig": _file("src/util.zig", "zig"),
        "src/util.c": _file("src/util.c", "c"),
    }
    graph = resolve_workspace_dependencies(parsed)
    assert graph.dependencies == ()
    assert graph.unresolved[0].reason == "ambiguous-relative-stem"
