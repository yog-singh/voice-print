"""Similarity scoring, score normalization and threshold calibration."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .numeric import matmul


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(matmul(l2_normalize(a.ravel()), l2_normalize(b.ravel())))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of a (n, d) and b (m, d)."""
    return matmul(l2_normalize(np.atleast_2d(a)), l2_normalize(np.atleast_2d(b)).T)


@dataclass
class ASNorm:
    """Adaptive symmetric score normalization against a cohort of background embeddings.

    Raw cosine scores drift with channel and recording conditions. AS-norm rescales a
    score by the statistics of each side against its most similar cohort speakers,
    which makes a single threshold transfer across microphones far better.
    """

    cohort: np.ndarray
    top_k: int = 300

    def __post_init__(self):
        self.cohort = l2_normalize(np.atleast_2d(np.asarray(self.cohort, dtype=np.float32)))
        self.top_k = min(self.top_k, len(self.cohort))

    def _stats(self, embedding: np.ndarray) -> Tuple[float, float]:
        scores = matmul(self.cohort, l2_normalize(embedding.ravel()))
        top = np.sort(scores)[-self.top_k :]
        return float(top.mean()), float(top.std() + 1e-9)

    def __call__(self, enroll: np.ndarray, test: np.ndarray, raw: Optional[float] = None) -> float:
        raw = cosine(enroll, test) if raw is None else raw
        mu_e, sd_e = self._stats(enroll)
        mu_t, sd_t = self._stats(test)
        return 0.5 * ((raw - mu_e) / sd_e + (raw - mu_t) / sd_t)


def equal_error_rate(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """Return (EER, threshold at the EER). labels: 1 for same speaker, 0 for different."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if not (labels == 1).any() or not (labels == 0).any():
        raise ValueError("EER needs at least one same-speaker and one different-speaker trial")

    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]

    n_target, n_nontarget = int((labels == 1).sum()), int((labels == 0).sum())
    # Sweeping the threshold downward: accepting the top i trials.
    false_accepts = np.concatenate([[0], np.cumsum(labels == 0)])
    false_rejects = n_target - np.concatenate([[0], np.cumsum(labels == 1)])

    far = false_accepts / n_nontarget
    frr = false_rejects / n_target
    idx = int(np.argmin(np.abs(far - frr)))
    # Index i means "accept the top i trials", so the operating point is the last accepted score.
    threshold = float(scores[idx - 1]) if idx > 0 else float(scores[0]) + 1e-6
    return float((far[idx] + frr[idx]) / 2), threshold


def calibrate_threshold(embeddings: np.ndarray, speaker_ids: Sequence[str]) -> dict:
    """Derive an operating threshold from labelled embeddings.

    Pass several utterances per speaker for at least a handful of speakers. A hardcoded
    threshold is meaningless across microphones and languages; this is the honest way.
    """
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    ids = np.asarray(speaker_ids)
    if len(embeddings) != len(ids):
        raise ValueError("embeddings and speaker_ids must have the same length")

    sim = matmul(embeddings, embeddings.T)
    iu = np.triu_indices(len(ids), k=1)
    scores = sim[iu]
    labels = (ids[iu[0]] == ids[iu[1]]).astype(np.int8)

    eer, threshold = equal_error_rate(scores, labels)
    return {
        "threshold": threshold,
        "eer": eer,
        "n_trials": int(len(scores)),
        "n_target": int(labels.sum()),
        "n_speakers": int(len(set(ids))),
    }
