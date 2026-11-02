"""Two processes persisting one project's index must not corrupt it.

The proxy, the MCP server and CLI commands started in the same project all
write `~/.entroly/checkpoints/<project>/index.json.gz`. Every writer staged
its bytes in the same `index.json.gz.tmp`, so a second writer truncated the
first one's staging file mid-write. Two processes writing 150 times each
produced 10 FileNotFoundErrors (one writer's rename found the file already
moved) and, out of 1,151 reads by a third process, 11 index files that would
not decompress or parse; a start at that moment logged "Failed to load
persistent index" and re-indexed from scratch.
"""

from __future__ import annotations

import gzip
import json
import os

from entroly.engine import EntrolyEngine


def _read(path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def test_a_write_interleaved_with_another_still_lands_whole(tmp_path, monkeypatch) -> None:
    """Another process completes a write between ours staging and renaming."""
    import entroly.engine as engine_module

    path = tmp_path / "index.json.gz"
    real_replace = os.replace
    other_writer_ran = []

    def replace(src, dst):
        if not other_writer_ran:
            other_writer_ran.append(True)
            EntrolyEngine._write_gzip_index(str(path), {"writer": "other process"})
        return real_replace(src, dst)

    monkeypatch.setattr(engine_module.os, "replace", replace)

    EntrolyEngine._write_gzip_index(str(path), {"writer": "this process"})

    assert _read(path) == {"writer": "this process"}
    assert not list(tmp_path.glob("*.tmp")), "no staging file left behind"


def test_normalising_a_plain_index_interleaved_with_a_write_lands_whole(
    tmp_path, monkeypatch
) -> None:
    import entroly.engine as engine_module

    path = tmp_path / "index.json.gz"
    path.write_text(json.dumps({"writer": "legacy plain json"}), encoding="utf-8")
    real_replace = os.replace
    other_writer_ran = []

    def replace(src, dst):
        if not other_writer_ran:
            other_writer_ran.append(True)
            EntrolyEngine._write_gzip_index(str(path), {"writer": "other process"})
        return real_replace(src, dst)

    monkeypatch.setattr(engine_module.os, "replace", replace)

    EntrolyEngine._ensure_gzip_index(str(path))

    assert _read(path) in ({"writer": "legacy plain json"}, {"writer": "other process"})
    assert not list(tmp_path.glob("*.tmp"))
