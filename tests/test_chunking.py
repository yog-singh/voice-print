from voicefingerprint.chunking import chunk_centers, compute_chunks, pad_to
from voicefingerprint.config import ChunkConfig

SR = 16000


def test_short_audio_yields_one_window():
    slices = compute_chunks(SR, SR, ChunkConfig(window_s=3.0))
    assert len(slices) == 1
    assert slices[0].stop == 3 * SR  # caller pads up to here


def test_windows_cover_the_whole_signal():
    n = 10 * SR
    slices = compute_chunks(n, SR, ChunkConfig(window_s=3.0, rate=1.0))
    assert slices[0].start == 0
    assert slices[-1].stop >= n


def test_higher_rate_gives_more_windows():
    n = 10 * SR
    sparse = compute_chunks(n, SR, ChunkConfig(window_s=3.0, rate=0.5))
    dense = compute_chunks(n, SR, ChunkConfig(window_s=3.0, rate=8.0))
    assert len(dense) > len(sparse)


def test_trailing_window_dropped_below_min_coverage():
    cfg = ChunkConfig(window_s=3.0, rate=1.0, min_coverage=0.9)
    n = int(4.1 * SR)
    slices = compute_chunks(n, SR, cfg)
    assert (n - slices[-1].start) / cfg.window_samples(SR) >= cfg.min_coverage


def test_step_never_exceeds_the_window():
    cfg = ChunkConfig(window_s=3.0, rate=0.01)  # absurdly low rate
    assert cfg.step_samples(SR) == cfg.window_samples(SR)


def test_pad_to_and_centers():
    import numpy as np

    assert len(pad_to(np.zeros(10), 25)) == 25
    assert len(pad_to(np.zeros(30), 25)) == 30
    centers = chunk_centers(compute_chunks(5 * SR, SR, ChunkConfig()), SR)
    assert (np.diff(centers) > 0).all()
