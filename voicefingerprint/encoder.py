"""ONNX speaker encoder. Waveforms in, L2-normalized embeddings out."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .audio import to_int16_scale
from .chunking import compute_chunks, pad_to
from .config import ChunkConfig, EncoderConfig
from .exceptions import ModelError
from .features import fbank, min_samples
from .logging_utils import get_logger
from .models import ModelSpec, get_spec, resolve_weights
from .scoring import l2_normalize

log = get_logger(__name__)


class SpeakerEncoder:
    def __init__(self, config: Optional[EncoderConfig] = None, chunk: Optional[ChunkConfig] = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ModelError("onnxruntime is required to run the encoder") from None

        self.config = config or EncoderConfig()
        self.chunk_config = chunk or ChunkConfig()
        self.spec: ModelSpec = get_spec(self.config.model)
        self.weights: Path = resolve_weights(self.spec, self.config.weights)

        options = ort.SessionOptions()
        if self.config.intra_op_threads:
            options.intra_op_num_threads = self.config.intra_op_threads
        providers = self.config.providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.weights), options, providers=providers)
        self._validate_signature()
        log.info("loaded %s (%s) on %s", self.spec.name, self.weights.name, self.session.get_providers()[0])

    def _validate_signature(self) -> None:
        inputs = {i.name: i.shape for i in self.session.get_inputs()}
        outputs = {o.name: o.shape for o in self.session.get_outputs()}
        if self.spec.input_name not in inputs:
            raise ModelError(f"model has inputs {list(inputs)}, expected {self.spec.input_name!r}")
        if self.spec.output_name not in outputs:
            raise ModelError(f"model has outputs {list(outputs)}, expected {self.spec.output_name!r}")

        n_mels = inputs[self.spec.input_name][-1]
        if isinstance(n_mels, int) and n_mels != self.spec.fbank.n_mels:
            raise ModelError(f"model wants {n_mels} mel bins, config says {self.spec.fbank.n_mels}")
        dim = outputs[self.spec.output_name][-1]
        if isinstance(dim, int) and dim != self.spec.embedding_dim:
            raise ModelError(f"model emits {dim}-d embeddings, registry says {self.spec.embedding_dim}")

    @property
    def sample_rate(self) -> int:
        return self.spec.fbank.sample_rate

    @property
    def embedding_dim(self) -> int:
        return self.spec.embedding_dim

    def features(self, wav: np.ndarray) -> np.ndarray:
        """Waveform -> (n_frames, n_mels) features in the exact form the model expects."""
        samples = to_int16_scale(wav) if self.spec.scale_to_int16 else wav
        return fbank(samples, self.spec.fbank)

    def _run(self, feats: np.ndarray) -> np.ndarray:
        if feats.ndim != 3:
            raise ModelError(f"expected a (batch, frames, mels) tensor, got shape {feats.shape}")
        outputs = self.session.run(
            [self.spec.output_name], {self.spec.input_name: np.ascontiguousarray(feats, dtype=np.float32)}
        )
        return l2_normalize(outputs[0])

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        """One embedding for one waveform, averaged over windows if the audio is long."""
        embeds, _ = self.embed_windows(wav)
        return l2_normalize(embeds.mean(axis=0))

    def embed_windows(self, wav: np.ndarray, chunk: Optional[ChunkConfig] = None) -> Tuple[np.ndarray, List[slice]]:
        """Per-window embeddings plus the sample slice each one came from."""
        chunk = chunk or self.chunk_config
        wav = np.asarray(wav, dtype=np.float32).ravel()
        floor = min_samples(self.spec.fbank)
        if len(wav) < floor:
            raise ModelError(f"audio is {len(wav)} samples, need at least {floor} ({floor / self.sample_rate:.2f}s)")

        slices = compute_chunks(len(wav), self.sample_rate, chunk)
        padded = pad_to(wav, slices[-1].stop)
        feats = np.stack([self.features(padded[s]) for s in slices])
        log.debug("embedding %d window(s) of %d frames", *feats.shape[:2])

        embeds = np.concatenate(
            [self._run(feats[i : i + self.config.batch_size]) for i in range(0, len(feats), self.config.batch_size)]
        )
        return embeds, slices

    def embed_timeline(self, wav: np.ndarray, rate: float = 4.0) -> Tuple[np.ndarray, List[slice]]:
        """Densely sampled embeddings for diarization. Higher rate = finer resolution, more RAM."""
        from dataclasses import replace

        return self.embed_windows(wav, replace(self.chunk_config, rate=rate))

    def embed_batch(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.embed_utterance(w) for w in wavs])

    def embed_speaker(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        """A single centroid embedding for several utterances of the same speaker."""
        if not wavs:
            raise ModelError("embed_speaker needs at least one waveform")
        return l2_normalize(self.embed_batch(wavs).mean(axis=0))

    def describe(self) -> dict:
        return {
            "model": self.spec.name,
            "weights": str(self.weights),
            "provider": self.session.get_providers()[0],
            "sample_rate": self.sample_rate,
            "embedding_dim": self.embedding_dim,
            "window_s": self.chunk_config.window_s,
            "default_threshold": self.spec.default_threshold,
            "notes": self.spec.notes,
        }
