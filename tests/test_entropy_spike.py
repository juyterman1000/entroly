"""Tests for Spec 2: Entropy-Spike Slicing — detect semantic boundaries."""

from __future__ import annotations

from entroly.entropy_spike import (
    EntropySpike,
    _shannon_entropy_py,
    detect_entropy_spikes,
    find_split_points,
)


def test_shannon_entropy_empty() -> None:
    assert _shannon_entropy_py("") == 0.0


def test_shannon_entropy_single_char() -> None:
    assert _shannon_entropy_py("aaaa") == 0.0


def test_shannon_entropy_uniform() -> None:
    h = _shannon_entropy_py("abcd")
    assert 1.9 < h < 2.1


def test_entropy_spike_dataclass() -> None:
    spike = EntropySpike(position=100, delta=0.5, left_entropy=3.0, right_entropy=3.5)
    assert spike.position == 100
    assert spike.delta == 0.5


def test_detect_spikes_short_text_returns_empty() -> None:
    assert detect_entropy_spikes("hello") == []


def test_detect_spikes_uniform_text_returns_empty() -> None:
    text = "word " * 200
    spikes = detect_entropy_spikes(text)
    assert len(spikes) == 0


def _make_transition_text() -> str:
    low_entropy = "aaa bbb ccc ddd eee fff ggg hhh " * 40
    high_entropy = "".join(
        chr(32 + (i * 7 + 13) % 95) for i in range(len(low_entropy))
    )
    return low_entropy + high_entropy


def test_detect_spikes_finds_transition() -> None:
    text = _make_transition_text()
    spikes = detect_entropy_spikes(text, spike_threshold=1.0)
    assert len(spikes) > 0
    transition_zone = len(text) // 2
    nearest = min(spikes, key=lambda s: abs(s.position - transition_zone))
    assert abs(nearest.position - transition_zone) < len(text) * 0.2


def test_detect_spikes_sorted_by_delta() -> None:
    text = _make_transition_text()
    spikes = detect_entropy_spikes(text, spike_threshold=1.0)
    if len(spikes) > 1:
        for i in range(len(spikes) - 1):
            assert spikes[i].delta >= spikes[i + 1].delta


def test_find_split_points_returns_sorted_positions() -> None:
    text = _make_transition_text()
    points = find_split_points(text, spike_threshold=1.0)
    assert points == sorted(points)


def test_find_split_points_empty_for_short_text() -> None:
    assert find_split_points("short text") == []


def test_find_split_points_respects_min_segment() -> None:
    text = _make_transition_text()
    points = find_split_points(text, min_segment_chars=500, spike_threshold=1.0)
    for i in range(len(points) - 1):
        assert points[i + 1] - points[i] >= 500


def test_find_split_points_respects_max_splits() -> None:
    text = _make_transition_text() * 3
    points = find_split_points(text, max_splits=2, spike_threshold=0.5)
    assert len(points) <= 2


def test_spike_threshold_sensitivity() -> None:
    text = _make_transition_text()
    loose = detect_entropy_spikes(text, spike_threshold=0.5)
    strict = detect_entropy_spikes(text, spike_threshold=3.0)
    assert len(loose) >= len(strict)


def test_ingest_uses_entropy_slicing(tmp_path: None) -> None:
    """Verify that the chunker imports and calls entropy_split_large_block."""
    from entroly.context_receipts.ingest import _entropy_split_large_block

    block_text = _make_transition_text()
    source_text = "prefix\n" + block_text + "\nsuffix"
    offset = len("prefix\n")
    block = {
        "text": block_text,
        "start": offset,
        "end": offset + len(block_text),
        "heading": None,
        "page": None,
    }

    chunks = _entropy_split_large_block(
        source_text,
        block,
        chunk_tokens=200,
        overlap_tokens=20,
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["start"] >= offset
        assert chunk["end"] <= offset + len(block_text)
        assert len(str(chunk["text"])) > 0
