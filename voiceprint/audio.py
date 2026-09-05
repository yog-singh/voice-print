from pathlib import Path
from typing import Optional, Union

import numpy as np

from .config import PreprocessConfig
from .exceptions import AudioError
from .logging_utils import get_logger

log = get_logger(__name__)

AudioInput = Union[str, Path, np.ndarray]
_INT16_MAX = 32767.0


def load_audio(source: AudioInput, target_sr: int, source_sr: Optional[int] = None) -> np.ndarray:
    """Load audio as mono float32 in [-1, 1] at target_sr."""
    if isinstance(source, (str, Path)):
        wav, source_sr = _read_file(source)
    else:
        wav = np.asarray(source)
        if source_sr is None:
            raise AudioError("source_sr is required when passing a raw array")

    if wav.ndim == 2:
        wav = wav.mean(axis=1 if wav.shape[0] > wav.shape[1] else 0)
    if wav.ndim != 1:
        raise AudioError(f"expected mono or stereo audio, got shape {wav.shape}")

    wav = wav.astype(np.float32, copy=False)
    if source_sr != target_sr:
        wav = resample(wav, source_sr, target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _read_file(path: Union[str, Path]):
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"audio file not found: {path}")
    try:
        import soundfile as sf
    except ImportError:
        raise AudioError("soundfile is required to read audio files") from None
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioError(f"could not decode {path}: {exc}") from exc
    return wav, sr


def resample(wav: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return wav
    try:
        import soxr
    except ImportError:
        raise AudioError(
            f"audio is {source_sr} Hz but the model needs {target_sr} Hz; install soxr to resample"
        ) from None
    return soxr.resample(wav, source_sr, target_sr).astype(np.float32)


def normalize_volume(wav: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale the waveform so its RMS sits at target_dbfs. Silent input is returned unchanged."""
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < 1e-8:
        log.warning("waveform is silent (rms=%.2e), skipping volume normalization", rms)
        return wav
    gain = 10 ** ((target_dbfs - 20 * np.log10(rms)) / 20)
    return (wav * gain).astype(np.float32)


def preprocess(
    source: AudioInput,
    config: Optional[PreprocessConfig] = None,
    source_sr: Optional[int] = None,
) -> np.ndarray:
    """Load, resample, level and (optionally) strip silence. Returns float32 mono."""
    config = config or PreprocessConfig()
    wav = load_audio(source, config.sample_rate, source_sr)
    original = len(wav)

    if config.normalize_volume:
        wav = normalize_volume(wav, config.target_dbfs)

    if config.trim_silence and config.vad_backend != "none":
        from .vad import get_vad, apply_vad

        wav = apply_vad(wav, get_vad(config.vad_backend, config.sample_rate), config.sample_rate)

    if len(wav) == 0:
        raise AudioError("no speech left after preprocessing")
    log.debug("preprocess: %d -> %d samples (%.0f%% kept)", original, len(wav), 100 * len(wav) / max(original, 1))
    return wav


def to_int16_scale(wav: np.ndarray) -> np.ndarray:
    return wav * _INT16_MAX
