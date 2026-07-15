from pathlib import Path

from scripts.check_docs import check_file, resolve_local_link


def test_resolve_local_link_handles_fragments_and_repo_site_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "docs" / "guide.md"
    source.parent.mkdir()
    source.write_text("# Guide\n", encoding="utf-8")

    assert resolve_local_link(source, "../README.md#install") == tmp_path / "README.md"
    assert resolve_local_link(source, "https://example.com/docs") is None
    assert resolve_local_link(source, "#quickstart") is None


def test_check_file_ignores_code_fences_and_reports_real_broken_links(
    tmp_path: Path,
) -> None:
    source = tmp_path / "README.md"
    existing = tmp_path / "guide.md"
    existing.write_text("# Guide\n", encoding="utf-8")
    source.write_text(
        "[works](guide.md)\n"
        "```markdown\n[example only](not-real.md)\n```\n"
        "[broken](missing.md)\n",
        encoding="utf-8",
    )

    broken = check_file(source)

    assert [(item.line, item.target) for item in broken] == [(5, "missing.md")]


def test_check_file_validates_html_assets(tmp_path: Path) -> None:
    source = tmp_path / "index.html"
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "logo.svg").write_text("<svg/>", encoding="utf-8")
    source.write_text(
        '<img src="assets/logo.svg"><a href="missing.html">Missing</a>',
        encoding="utf-8",
    )

    broken = check_file(source)

    assert len(broken) == 1
    assert broken[0].target == "missing.html"
