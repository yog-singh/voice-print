"""Split long audio into fixed windows so every embedding sees a comparable span."""

from typing import List

import numpy as np

from .config import ChunkConfig


def compute_chunks(n_samples: int, sample_rate: int, config: ChunkConfig) -> List[slice]:
    """Return overlapping sample slices covering the waveform.

    The final slice may run past n_samples; the caller is expected to zero-pad up to
    slices[-1].stop. A trailing window is dropped when less than min_coverage of it
    holds real audio, unless it is the only window.
    """
    window = config.window_samples(sample_rate)
    step = config.step_samples(sample_rate)

    starts = list(range(0, max(1, n_samples - window + step), step))
    slices = [slice(s, s + window) for s in starts]

    if len(slices) > 1:
        coverage = (n_samples - slices[-1].start) / window
        if coverage < config.min_coverage:
            slices.pop()
    return slices


def pad_to(wav: np.ndarray, length: int) -> np.ndarray:
    if len(wav) >= length:
        return wav
    return np.pad(wav, (0, length - len(wav)))


def chunk_centers(slices: List[slice], sample_rate: int) -> np.ndarray:
    """Midpoint of each slice in seconds, useful for plotting a diarization timeline."""
    return np.array([(s.start + s.stop) / 2 / sample_rate for s in slices])
