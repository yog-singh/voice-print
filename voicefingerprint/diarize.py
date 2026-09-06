"""Who spoke when. Dense embeddings over the timeline, then clustering or enrolled matching."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .logging_utils import get_logger
from .numeric import matmul
from .scoring import l2_normalize

log = get_logger(__name__)


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    score: float

    def __repr__(self) -> str:
        return f"Turn({self.speaker!r}, {self.start:.2f}-{self.end:.2f}s, score={self.score:.3f})"


def agglomerative(embeddings: np.ndarray, threshold: float = 0.55, n_clusters: Optional[int] = None) -> np.ndarray:
    """Average-linkage clustering on cosine similarity.

    Merges the closest pair until either n_clusters remain or the best similarity drops
    below threshold. O(n^2) memory, which is fine for a timeline of a few thousand windows.
    """
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)

    sim = matmul(embeddings, embeddings.T)
    np.fill_diagonal(sim, -np.inf)
    sizes = np.ones(n)
    active = np.ones(n, dtype=bool)
    labels = np.arange(n)

    while active.sum() > 1:
        if n_clusters is not None and active.sum() <= n_clusters:
            break
        masked = np.where(active[:, None] & active[None, :], sim, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        if n_clusters is None and masked[i, j] < threshold:
            break

        # Average linkage: the merged cluster's similarity is the size-weighted mean.
        sim[i] = (sim[i] * sizes[i] + sim[j] * sizes[j]) / (sizes[i] + sizes[j])
        sim[:, i] = sim[i]
        sim[i, i] = -np.inf
        sizes[i] += sizes[j]
        active[j] = False
        labels[labels == j] = i

    _, compact = np.unique(labels, return_inverse=True)
    return compact


def _turns_from_windows(labels, slices, scores, sample_rate: int, n_samples: int) -> List[Turn]:
    """Turn per-window labels into a contiguous timeline.

    Windows overlap, so a boundary is placed halfway between neighbouring window
    centres rather than at the window edges. Otherwise adjacent turns overlap.
    """
    if len(slices) == 0:
        return []
    centers = np.array([(s.start + s.stop) / 2 for s in slices], dtype=np.float64)
    edges = np.concatenate([[slices[0].start], (centers[:-1] + centers[1:]) / 2, [min(slices[-1].stop, n_samples)]])
    edges = np.clip(edges, 0, n_samples)
    return [
        Turn(str(label), float(edges[i] / sample_rate), float(edges[i + 1] / sample_rate), float(score))
        for i, (label, score) in enumerate(zip(labels, scores))
    ]


def _merge(turns: List[Turn], min_duration: float) -> List[Turn]:
    merged: List[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker and turn.start <= merged[-1].end + 1e-6:
            previous = merged[-1]
            merged[-1] = Turn(
                previous.speaker, previous.start, max(previous.end, turn.end), max(previous.score, turn.score)
            )
        else:
            merged.append(turn)
    return [t for t in merged if t.end - t.start >= min_duration]


def diarize(
    recognizer,
    source,
    rate: float = 4.0,
    n_speakers: Optional[int] = None,
    threshold: float = 0.55,
    min_duration: float = 0.5,
) -> List[Turn]:
    """Unsupervised diarization: cluster the timeline into "speaker_0", "speaker_1", ..."""
    wav = recognizer.preprocess(source)
    embeddings, slices = recognizer.encoder.embed_timeline(wav, rate=rate)
    labels = agglomerative(embeddings, threshold=threshold, n_clusters=n_speakers)
    log.info("diarized %d windows into %d speaker(s)", len(labels), len(set(labels)))

    turns = _turns_from_windows(
        [f"speaker_{label}" for label in labels],
        slices,
        np.ones(len(labels)),
        recognizer.encoder.sample_rate,
        len(wav),
    )
    return _merge(turns, min_duration)


def assign_to_enrolled(
    recognizer,
    source,
    rate: float = 4.0,
    threshold: Optional[float] = None,
    min_duration: float = 0.5,
    unknown_label: str = "unknown",
) -> List[Turn]:
    """Diarization against known speakers: label each window with the best enrolled match."""
    names, centroids = recognizer.book.centroids()
    if not names:
        raise ValueError("no speakers enrolled; use diarize() for the unsupervised case")
    threshold = recognizer.threshold if threshold is None else threshold

    wav = recognizer.preprocess(source)
    embeddings, slices = recognizer.encoder.embed_timeline(wav, rate=rate)
    scores = matmul(embeddings, centroids.T)
    best = np.argmax(scores, axis=1)

    best_scores = scores[np.arange(len(best)), best]
    assigned = [names[b] if sc >= threshold else unknown_label for b, sc in zip(best, best_scores)]
    turns = _turns_from_windows(assigned, slices, best_scores, recognizer.encoder.sample_rate, len(wav))
    return _merge(turns, min_duration)
