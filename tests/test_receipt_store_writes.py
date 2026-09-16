"""Artifact writes preserve prior evidence on failure and restrict access."""
import os
import stat

import pytest

from entroly.context_receipts import store


def test_json_roundtrip_preserves_unicode_and_replaces_existing_file(tmp_path):
    target = tmp_path / "receipts" / "index.json"
    store.write_json(target, {"source": "old"})
    data = {"source": "exact\r\nα\n", "digest": "123"}
    store.write_json(target, data)
    assert store.read_json(target) == data
    assert list(target.parent.iterdir()) == [target]


def test_failed_replace_preserves_previous_receipt_and_removes_temporary(tmp_path, monkeypatch):
    target = tmp_path / "receipt.json"
    store.write_json(target, {"original": True})
    previous = target.read_bytes()

    def fail(*args):
        raise PermissionError("simulated locked destination")

    monkeypatch.setattr(store.os, "replace", fail)
    with pytest.raises(PermissionError):
        store.write_json(target, {"original": False})
    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes; Windows uses inherited ACLs")
def test_artifact_modes_are_private_even_with_permissive_umask(tmp_path):
    old = os.umask(0)
    try:
        directory = store.ensure_store(tmp_path / "receipts")
        target = store.write_json(directory / "receipt.json", {"sensitive": "source"})
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        target.chmod(0o666)
        store.write_text(target, "replacement")
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    finally:
        os.umask(old)


def test_symlink_destination_is_rejected(tmp_path):
    original = tmp_path / "source.txt"
    original.write_text("keep", encoding="utf-8")
    link = tmp_path / "receipt.txt"
    try:
        link.symlink_to(original)
    except OSError:
        pytest.skip("host does not permit symlinks")
    with pytest.raises(ValueError, match="symlink"):
        store.write_text(link, "overwrite")
    assert original.read_text(encoding="utf-8") == "keep"
