import numpy as np
import pytest

from voicefingerprint.diarize import Turn, _merge, agglomerative


def _clusters(n_clusters, per_cluster, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    return np.concatenate([c + 0.03 * rng.standard_normal((per_cluster, dim)) for c in centers])


def test_recovers_known_cluster_count():
    labels = agglomerative(_clusters(3, 10), n_clusters=3)
    assert len(set(labels)) == 3
    for start in (0, 10, 20):
        assert len(set(labels[start : start + 10])) == 1


def test_threshold_mode_finds_structure():
    labels = agglomerative(_clusters(2, 12), threshold=0.9)
    assert 1 < len(set(labels)) <= 4


def test_labels_are_compact():
    labels = agglomerative(_clusters(4, 5), n_clusters=4)
    assert sorted(set(labels)) == list(range(len(set(labels))))


def test_empty_input():
    assert len(agglomerative(np.zeros((0, 16)))) == 0


def test_merge_joins_adjacent_and_drops_short_turns():
    turns = [
        Turn("a", 0.0, 1.0, 0.9),
        Turn("a", 1.0, 2.0, 0.8),
        Turn("b", 2.0, 2.1, 0.7),
        Turn("a", 2.1, 4.0, 0.6),
    ]
    merged = _merge(turns, min_duration=0.5)
    assert [t.speaker for t in merged] == ["a", "a"]
    assert merged[0].start == 0.0 and merged[0].end == 2.0


def test_turns_from_windows_are_contiguous_and_non_overlapping():
    from voicefingerprint.diarize import _turns_from_windows

    sr = 16000
    slices = [slice(i * sr, i * sr + 3 * sr) for i in range(5)]
    labels = ["a", "a", "b", "b", "b"]
    turns = _turns_from_windows(labels, slices, np.ones(5), sr, 7 * sr)

    assert turns[0].start == 0.0
    assert turns[-1].end == 7.0
    for previous, following in zip(turns, turns[1:]):
        assert previous.end == following.start


def test_turns_clamp_to_the_waveform_length():
    from voicefingerprint.diarize import _turns_from_windows

    sr = 16000
    slices = [slice(0, 3 * sr), slice(2 * sr, 5 * sr)]
    turns = _turns_from_windows(["a", "b"], slices, np.ones(2), sr, int(4.2 * sr))
    assert turns[-1].end == pytest.approx(4.2)
