"""Persistent enrollment store: named speakers and their embedding centroids."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .exceptions import EnrollmentError
from .logging_utils import get_logger
from .numeric import matmul
from .scoring import l2_normalize

log = get_logger(__name__)

FORMAT_VERSION = 1


@dataclass
class Speaker:
    name: str
    embeddings: np.ndarray  # (n_utterances, dim), each L2-normalized
    metadata: dict = field(default_factory=dict)

    @property
    def centroid(self) -> np.ndarray:
        return l2_normalize(self.embeddings.mean(axis=0))

    @property
    def n_utterances(self) -> int:
        return len(self.embeddings)

    def self_consistency(self) -> float:
        """Mean pairwise similarity of the enrolled utterances. Low values mean noisy enrollment."""
        if self.n_utterances < 2:
            return float("nan")
        sim = matmul(self.embeddings, self.embeddings.T)
        iu = np.triu_indices(self.n_utterances, k=1)
        return float(sim[iu].mean())


class SpeakerBook:
    """A small in-memory database of enrolled speakers, serializable to a single .npz file."""

    def __init__(self, model: str, embedding_dim: int):
        self.model = model
        self.embedding_dim = embedding_dim
        self._speakers: Dict[str, Speaker] = {}

    def __len__(self) -> int:
        return len(self._speakers)

    def __contains__(self, name: str) -> bool:
        return name in self._speakers

    def __getitem__(self, name: str) -> Speaker:
        try:
            return self._speakers[name]
        except KeyError:
            raise EnrollmentError(f"speaker {name!r} is not enrolled; known: {self.names}") from None

    @property
    def names(self) -> List[str]:
        return sorted(self._speakers)

    def add(self, name: str, embeddings: np.ndarray, metadata: Optional[dict] = None) -> Speaker:
        """Enroll a speaker, or append to an existing enrollment."""
        embeddings = l2_normalize(np.atleast_2d(np.asarray(embeddings, dtype=np.float32)))
        if embeddings.shape[1] != self.embedding_dim:
            raise EnrollmentError(f"expected {self.embedding_dim}-d embeddings, got {embeddings.shape[1]}-d")

        if name in self._speakers:
            existing = self._speakers[name]
            existing.embeddings = np.concatenate([existing.embeddings, embeddings])
            existing.metadata.update(metadata or {})
        else:
            self._speakers[name] = Speaker(name, embeddings, dict(metadata or {}))
        log.info("enrolled %r with %d utterance(s) total", name, self._speakers[name].n_utterances)
        return self._speakers[name]

    def remove(self, name: str) -> None:
        if self._speakers.pop(name, None) is None:
            raise EnrollmentError(f"speaker {name!r} is not enrolled")

    def centroids(self) -> Tuple[List[str], np.ndarray]:
        """Names and their (n_speakers, dim) centroid matrix, in a stable order."""
        if not self._speakers:
            return [], np.zeros((0, self.embedding_dim), dtype=np.float32)
        names = self.names
        return names, np.stack([self._speakers[n].centroid for n in names])

    def all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Every enrolled embedding with its speaker label. Feeds calibrate_threshold."""
        rows, labels = [], []
        for name in self.names:
            speaker = self._speakers[name]
            rows.append(speaker.embeddings)
            labels.extend([name] * speaker.n_utterances)
        if not rows:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []
        return np.concatenate(rows), labels

    def save(self, path) -> Path:
        path = Path(path).with_suffix(".npz")
        arrays, index = {}, []
        for i, name in enumerate(self.names):
            speaker = self._speakers[name]
            arrays[f"emb_{i}"] = speaker.embeddings
            index.append({"name": name, "key": f"emb_{i}", "metadata": speaker.metadata})

        header = json.dumps(
            {"version": FORMAT_VERSION, "model": self.model, "embedding_dim": self.embedding_dim, "speakers": index}
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, header=np.array(header), **arrays)
        log.info("saved %d speaker(s) to %s", len(self), path)
        return path

    @classmethod
    def load(cls, path, expect_model: Optional[str] = None) -> "SpeakerBook":
        path = Path(path).with_suffix(".npz")
        if not path.is_file():
            raise EnrollmentError(f"speaker book not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            if header.get("version") != FORMAT_VERSION:
                raise EnrollmentError(f"unsupported speaker book version {header.get('version')}")
            # Embeddings from different encoders are not comparable, so refuse to mix them.
            if expect_model and header["model"] != expect_model:
                raise EnrollmentError(
                    f"{path} was built with model {header['model']!r} but the encoder is {expect_model!r}"
                )
            book = cls(header["model"], header["embedding_dim"])
            for entry in header["speakers"]:
                book.add(entry["name"], data[entry["key"]], entry.get("metadata"))
        return book

    def summary(self) -> List[dict]:
        return [
            {
                "name": name,
                "utterances": self._speakers[name].n_utterances,
                "self_consistency": round(self._speakers[name].self_consistency(), 4),
            }
            for name in self.names
        ]
