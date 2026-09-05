"""Kaldi-compatible log-mel filterbank features.

Reimplemented in NumPy so the library needs no PyTorch/torchaudio at runtime. The
arithmetic mirrors torchaudio.compliance.kaldi.fbank step for step; see
tests/test_features.py, which asserts parity when torchaudio is installed.
"""

from typing import Optional

import numpy as np

from .config import FbankConfig
from .exceptions import FeatureError
from .numeric import matmul

_EPSILON = float(np.finfo(np.float32).eps)


def _mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _window(name: str, size: int) -> np.ndarray:
    n = np.arange(size, dtype=np.float64)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (size - 1))
    if name == "hanning":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (size - 1))
    if name == "povey":
        return (0.5 - 0.5 * np.cos(2 * np.pi * n / (size - 1))) ** 0.85
    if name == "rectangular":
        return np.ones(size, dtype=np.float64)
    raise FeatureError(f"unsupported window type {name!r}")


def mel_filterbank(cfg: FbankConfig) -> np.ndarray:
    """Triangular mel weights of shape (n_mels, padded_length // 2 + 1)."""
    n_fft_bins = cfg.padded_length // 2
    nyquist = 0.5 * cfg.sample_rate
    high_freq = cfg.high_freq if cfg.high_freq > 0 else nyquist + cfg.high_freq
    if not 0 <= cfg.low_freq < high_freq <= nyquist:
        raise FeatureError(f"bad frequency range: [{cfg.low_freq}, {high_freq}] for {nyquist} Hz Nyquist")

    fft_bin_width = cfg.sample_rate / cfg.padded_length
    mel_low, mel_high = _mel_scale(np.array([cfg.low_freq])), _mel_scale(np.array([high_freq]))
    delta = (mel_high - mel_low) / (cfg.n_mels + 1)

    bins = np.arange(cfg.n_mels, dtype=np.float64)[:, None]
    left = mel_low + bins * delta
    center = mel_low + (bins + 1) * delta
    right = mel_low + (bins + 2) * delta

    mel = _mel_scale(fft_bin_width * np.arange(n_fft_bins, dtype=np.float64))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    weights = np.maximum(0.0, np.minimum(up, down))

    # The Nyquist bin carries no weight but the spectrum has it, so pad to match.
    return np.pad(weights, ((0, 0), (0, 1)))


def _frame(wav: np.ndarray, size: int, shift: int, snip_edges: bool) -> np.ndarray:
    if snip_edges:
        if len(wav) < size:
            raise FeatureError(
                f"audio is {len(wav)} samples but one frame needs {size}; "
                "supply a longer clip or lower frame_length_ms"
            )
        n_frames = 1 + (len(wav) - size) // shift
    else:
        n_frames = (len(wav) + shift // 2) // shift
        pad = (n_frames - 1) * shift + size - len(wav)
        if pad > 0:
            wav = np.pad(wav, (size // 2, pad), mode="reflect")

    strides = (wav.strides[0] * shift, wav.strides[0])
    return np.lib.stride_tricks.as_strided(wav, (n_frames, size), strides).copy()


def fbank(wav: np.ndarray, cfg: Optional[FbankConfig] = None) -> np.ndarray:
    """Compute log-mel filterbank features of shape (n_frames, n_mels)."""
    cfg = cfg or FbankConfig()
    wav = np.asarray(wav, dtype=np.float64).ravel()

    frames = _frame(wav, cfg.frame_length, cfg.frame_shift, cfg.snip_edges)

    if cfg.dither > 0:
        frames = frames + cfg.dither * np.random.randn(*frames.shape)
    if cfg.remove_dc_offset:
        frames = frames - frames.mean(axis=1, keepdims=True)
    if cfg.preemphasis != 0.0:
        shifted = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        frames = frames - cfg.preemphasis * shifted

    frames = frames * _window(cfg.window, cfg.frame_length)[None, :]
    if cfg.padded_length > cfg.frame_length:
        frames = np.pad(frames, ((0, 0), (0, cfg.padded_length - cfg.frame_length)))

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energies = matmul(power, mel_filterbank(cfg).T)
    feats = np.log(np.maximum(energies, _EPSILON))

    if cfg.cmn:
        feats = feats - feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


def min_samples(cfg: FbankConfig, n_frames: int = 1) -> int:
    """Shortest waveform that yields n_frames frames."""
    return cfg.frame_length + (n_frames - 1) * cfg.frame_shift
