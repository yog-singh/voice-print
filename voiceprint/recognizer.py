"""High-level API: enroll speakers, verify a pair, identify against the book."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .audio import AudioInput, preprocess
from .config import Config
from .encoder import SpeakerEncoder
from .exceptions import EnrollmentError
from .logging_utils import get_logger
from .scoring import ASNorm, calibrate_threshold, cosine, cosine_matrix
from .store import SpeakerBook

log = get_logger(__name__)


@dataclass
class Match:
    name: str
    score: float
    accepted: bool


@dataclass
class VerificationResult:
    score: float
    threshold: float
    accepted: bool
    normalized: bool

    def __bool__(self) -> bool:
        return self.accepted


class VoiceRecognizer:
    """Ties together preprocessing, the encoder and an enrollment store.

    Note: this answers "does this voice match the enrolled one", not "is a live human
    speaking". A replayed recording of an enrolled speaker will pass. Anti-spoofing is a
    separate problem and deliberately not folded into verify().
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        book: Optional[SpeakerBook] = None,
        threshold: Optional[float] = None,
        cohort: Optional[np.ndarray] = None,
    ):
        self.config = config or Config()
        self.encoder = SpeakerEncoder(self.config.encoder, self.config.chunk)
        self.book = book or SpeakerBook(self.encoder.spec.name, self.encoder.embedding_dim)
        self.threshold = self.encoder.spec.default_threshold if threshold is None else threshold
        self.asnorm = ASNorm(cohort) if cohort is not None else None

        if self.book.model != self.encoder.spec.name:
            raise EnrollmentError(
                f"speaker book was built with {self.book.model!r} but the encoder is {self.encoder.spec.name!r}"
            )

    def preprocess(self, source: AudioInput, source_sr: Optional[int] = None) -> np.ndarray:
        return preprocess(source, self.config.preprocess, source_sr)

    def embed(self, source: AudioInput, source_sr: Optional[int] = None) -> np.ndarray:
        """Embedding for one audio file, array or preprocessed waveform."""
        return self.encoder.embed_utterance(self.preprocess(source, source_sr))

    def embed_many(self, sources: Sequence[AudioInput], source_sr: Optional[int] = None) -> np.ndarray:
        return np.stack([self.embed(s, source_sr) for s in sources])

    def enroll(self, name: str, sources: Sequence[AudioInput], source_sr: Optional[int] = None, **metadata):
        """Enroll a speaker from several clips. Three or more varied clips beats one long one."""
        if not sources:
            raise EnrollmentError(f"no audio supplied for {name!r}")
        embeddings = self.embed_many(sources, source_sr)
        speaker = self.book.add(name, embeddings, metadata or None)
        if speaker.n_utterances >= 2 and speaker.self_consistency() < 0.5:
            log.warning(
                "enrollment for %r is inconsistent (%.2f); the clips may contain different speakers",
                name,
                speaker.self_consistency(),
            )
        return speaker

    def score(self, a: AudioInput, b: AudioInput, source_sr: Optional[int] = None) -> float:
        return self.score_embeddings(self.embed(a, source_sr), self.embed(b, source_sr))

    def score_embeddings(self, a: np.ndarray, b: np.ndarray) -> float:
        raw = cosine(a, b)
        return self.asnorm(a, b, raw) if self.asnorm else raw

    def verify(
        self,
        a: AudioInput,
        b: AudioInput,
        threshold: Optional[float] = None,
        source_sr: Optional[int] = None,
    ) -> VerificationResult:
        threshold = self.threshold if threshold is None else threshold
        score = self.score(a, b, source_sr)
        return VerificationResult(score, threshold, score >= threshold, self.asnorm is not None)

    def verify_speaker(
        self, name: str, source: AudioInput, threshold: Optional[float] = None, source_sr: Optional[int] = None
    ) -> VerificationResult:
        """Check a clip against one enrolled speaker."""
        threshold = self.threshold if threshold is None else threshold
        score = self.score_embeddings(self.book[name].centroid, self.embed(source, source_sr))
        return VerificationResult(score, threshold, score >= threshold, self.asnorm is not None)

    def identify(
        self,
        source: AudioInput,
        top_k: int = 3,
        threshold: Optional[float] = None,
        source_sr: Optional[int] = None,
    ) -> List[Match]:
        """Rank enrolled speakers against a clip. An empty accepted set means 'unknown'."""
        if len(self.book) == 0:
            raise EnrollmentError("no speakers enrolled")
        threshold = self.threshold if threshold is None else threshold

        embedding = self.embed(source, source_sr)
        names, centroids = self.book.centroids()
        if self.asnorm:
            scores = np.array([self.asnorm(c, embedding) for c in centroids])
        else:
            scores = cosine_matrix(embedding[None, :], centroids)[0]

        order = np.argsort(-scores)[:top_k]
        return [Match(names[i], float(scores[i]), bool(scores[i] >= threshold)) for i in order]

    def calibrate(self) -> dict:
        """Fit a threshold on the enrolled data and adopt it. Needs >= 2 speakers."""
        embeddings, labels = self.book.all_embeddings()
        if len(set(labels)) < 2:
            raise EnrollmentError("calibration needs at least two enrolled speakers")
        report = calibrate_threshold(embeddings, labels)
        self.threshold = report["threshold"]
        log.info("calibrated threshold=%.4f (EER %.2f%%)", report["threshold"], 100 * report["eer"])
        return report

    def set_cohort(self, sources: Sequence[AudioInput]) -> None:
        """Enable AS-norm using background audio from speakers you do NOT want to recognize."""
        self.asnorm = ASNorm(self.embed_many(sources))

    def save(self, path: Union[str, Path]) -> Path:
        return self.book.save(path)

    def load(self, path: Union[str, Path]) -> SpeakerBook:
        self.book = SpeakerBook.load(path, expect_model=self.encoder.spec.name)
        return self.book
