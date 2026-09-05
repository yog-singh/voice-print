"""Voice activity detection used to strip non-speech before embedding."""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

from .exceptions import ModelError
from .logging_utils import get_logger

log = get_logger(__name__)

Segment = Tuple[int, int]  # [start, end) in samples


class VAD(Protocol):
    def speech_mask(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a per-sample boolean mask marking speech."""


def _smooth(flags: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return flags
    kernel = np.ones(width) / width
    return np.convolve(flags.astype(np.float64), kernel, mode="same")


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return mask
    padded = np.pad(mask.astype(np.float64), width)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * width + 1)
    return windows.max(axis=1).astype(bool)


@dataclass
class EnergyVAD:
    """Adaptive log-energy gate. No extra dependencies; good enough for clean speech."""

    frame_ms: float = 30.0
    threshold_db: float = 25.0  # how far below the loud percentile counts as silence
    smoothing_frames: int = 8
    dilation_frames: int = 6

    def speech_mask(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        frame = int(sample_rate * self.frame_ms / 1000)
        n_frames = len(wav) // frame
        if n_frames == 0:
            return np.ones(len(wav), dtype=bool)

        frames = wav[: n_frames * frame].reshape(n_frames, frame)
        energy = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
        flags = energy > (np.percentile(energy, 95) - self.threshold_db)

        mask = _dilate(np.round(_smooth(flags, self.smoothing_frames)).astype(bool), self.dilation_frames)
        mask = np.repeat(mask, frame)
        return np.pad(mask, (0, len(wav) - len(mask)), constant_values=mask[-1] if len(mask) else True)


class SileroVAD:
    """Silero VAD v4/v5 via onnxruntime. More robust than EnergyVAD in noise."""

    REPO = "onnx-community/silero-vad"
    FILE = "onnx/model.onnx"
    WINDOW = 512  # samples, fixed by the model at 16 kHz

    def __init__(self, threshold: float = 0.5, weights: Optional[str] = None, dilation_frames: int = 6):
        import onnxruntime as ort

        path = weights or self._fetch()
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.threshold = threshold
        self.dilation_frames = dilation_frames

    @classmethod
    def _fetch(cls) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ModelError("huggingface_hub is required to download the Silero VAD model") from None
        try:
            return hf_hub_download(cls.REPO, cls.FILE)
        except Exception as exc:
            raise ModelError(f"could not fetch Silero VAD: {exc}") from exc

    def _initial_state(self):
        state = {}
        for name, meta in self.inputs.items():
            if name in ("input", "sr"):
                continue
            # Dynamic axes come back as None or as a symbolic name; batch is always 1 here.
            shape = [d if isinstance(d, int) else 1 for d in meta.shape]
            state[name] = np.zeros(shape, dtype=np.float32)
        return state

    def speech_mask(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate not in (8000, 16000):
            raise ModelError(f"Silero VAD supports 8 or 16 kHz, got {sample_rate}")

        n_windows = len(wav) // self.WINDOW
        if n_windows == 0:
            return np.ones(len(wav), dtype=bool)

        state = self._initial_state()
        probs = np.empty(n_windows, dtype=np.float32)
        for i in range(n_windows):
            chunk = wav[i * self.WINDOW : (i + 1) * self.WINDOW].astype(np.float32)[None, :]
            feed = {"input": chunk, **state}
            if "sr" in self.inputs:
                feed["sr"] = np.array(sample_rate, dtype=np.int64)
            outputs = self.session.run(None, feed)
            probs[i] = float(np.ravel(outputs[0])[0])
            state = self._carry_state(outputs)

        flags = _dilate(probs > self.threshold, self.dilation_frames)
        mask = np.repeat(flags, self.WINDOW)
        return np.pad(mask, (0, len(wav) - len(mask)), constant_values=True)

    def _carry_state(self, outputs) -> dict:
        names = [n for n in self.inputs if n not in ("input", "sr")]
        return {name: outputs[i + 1] for i, name in enumerate(names)}


def get_vad(backend: str, sample_rate: int = 16000) -> Optional[VAD]:
    if backend == "none":
        return None
    if backend == "energy":
        return EnergyVAD()
    if backend == "silero":
        return SileroVAD()
    raise ValueError(f"unknown VAD backend {backend!r}; expected 'energy', 'silero' or 'none'")


def apply_vad(wav: np.ndarray, vad: Optional[VAD], sample_rate: int, min_keep_ratio: float = 0.1) -> np.ndarray:
    """Drop non-speech samples, but keep the original if the VAD removed almost everything."""
    if vad is None:
        return wav
    mask = vad.speech_mask(wav, sample_rate)
    kept = wav[mask]
    if len(kept) < min_keep_ratio * len(wav):
        log.warning("VAD kept only %.1f%% of the audio; falling back to the untrimmed waveform", 100 * len(kept) / max(len(wav), 1))
        return wav
    return np.ascontiguousarray(kept)


def mask_to_segments(mask: np.ndarray) -> List[Segment]:
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))
