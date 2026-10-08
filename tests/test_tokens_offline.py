"""A tokenizer cache miss/corruption must not authorize a network request."""

import hashlib
import pytest

from entroly.tokens import _encoding, count_tokens


@pytest.fixture
def isolated_encoder(monkeypatch, tmp_path):
    pytest.importorskip("tiktoken")
    from tiktoken import load, registry

    _encoding.cache_clear()
    monkeypatch.setattr(registry, "ENCODINGS", {})
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("canonical offline counting attempted an asset fetch")

    monkeypatch.setattr(load, "read_file_cached", forbidden)
    yield tmp_path
    _encoding.cache_clear()


def test_missing_tokenizer_asset_stays_unavailable_without_fetch(isolated_encoder):
    assert _encoding() is None
    assert count_tokens("abcd") == 1
    assert list(isolated_encoder.iterdir()) == []


def test_corrupt_tokenizer_asset_is_preserved_and_never_refetched(isolated_encoder):
    from tiktoken_ext.openai_public import o200k_base

    url = next(
        x
        for x in o200k_base.__code__.co_consts
        if isinstance(x, str) and x.startswith("https://")
    )
    path = isolated_encoder / hashlib.sha1(url.encode()).hexdigest()
    path.write_bytes(b"corrupt local asset")
    assert _encoding() is None
    assert path.read_bytes() == b"corrupt local asset"


def test_offline_constructor_matches_the_canonical_installed_encoder(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    from tiktoken import load, registry

    _encoding.cache_clear()
    encoder = _encoding()
    if encoder is None:
        pytest.skip("verified local tokenizer asset unavailable")
    # Comparing to the public canonical implementation is safe only after the
    # local asset has been verified; this test is an explicit setup consumer.
    canonical = tiktoken.get_encoding("o200k_base")
    monkeypatch.setattr(registry, "ENCODINGS", {})
    monkeypatch.setattr(
        load,
        "read_file_cached",
        lambda *a, **kw: pytest.fail("remote-capable reader invoked"),
    )
    _encoding.cache_clear()
    offline = _encoding()
    assert offline is not None
    for text in (
        "naive café 日本語",
        "def f(x):\n    return x + 7\n",
        "alpha\r\nbeta",
        "🧠 decision evidence",
    ):
        assert offline.encode(text) == canonical.encode(text)
    _encoding.cache_clear()
