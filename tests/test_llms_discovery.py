from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_llms_file_matches_legacy_mirror() -> None:
    canonical = (ROOT / "llms.txt").read_text(encoding="utf-8")
    mirror = (ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")

    assert canonical == mirror
    assert canonical.startswith("# Entroly\n\n> ")


def test_llms_files_keep_trust_boundaries_explicit() -> None:
    canonical = (ROOT / "llms.txt").read_text(encoding="utf-8")
    extended = (ROOT / "docs" / "llms-full.txt").read_text(encoding="utf-8")
    combined = canonical + extended

    required = (
        "Proxy mode still sends the selected prompt",
        "does not establish universal truth",
        "workload",
        "https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md",
    )
    for phrase in required:
        assert phrase in combined

    forbidden = (
        "Your code never leaves your machine",
        "discounts actually apply",
        "No. Benchmarks show",
        "entroly/stave.py",
    )
    for phrase in forbidden:
        assert phrase not in combined


def test_robots_points_to_canonical_llms_file() -> None:
    robots = (ROOT / "docs" / "robots.txt").read_text(encoding="utf-8")

    assert "https://juyterman1000.github.io/entroly/llms.txt" in robots
