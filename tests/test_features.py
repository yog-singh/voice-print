import numpy as np
import pytest

from voicefingerprint.config import FbankConfig
from voicefingerprint.exceptions import FeatureError
from voicefingerprint.features import fbank, mel_filterbank, min_samples

CFG = FbankConfig(cmn=False)


def test_shape_and_frame_count(wav_a):
    feats = fbank(wav_a, CFG)
    expected = 1 + (len(wav_a) - CFG.frame_length) // CFG.frame_shift
    assert feats.shape == (expected, 80)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_filterbank_is_triangular_and_covers_the_band():
    weights = mel_filterbank(CFG)
    assert weights.shape == (80, CFG.padded_length // 2 + 1)
    assert (weights >= 0).all()
    assert weights[:, -1].sum() == 0  # the Nyquist bin is padding only
    assert (weights.sum(axis=1) > 0).all()


def test_cmn_zeroes_the_time_mean(wav_a):
    feats = fbank(wav_a, FbankConfig(cmn=True))
    assert np.allclose(feats.mean(axis=0), 0, atol=1e-4)


def test_too_short_audio_raises():
    with pytest.raises(FeatureError, match="one frame needs"):
        fbank(np.zeros(100, dtype=np.float32), CFG)


def test_min_samples_is_the_exact_boundary():
    for n_frames in (1, 5, 50):
        wav = np.random.randn(min_samples(CFG, n_frames)).astype(np.float32)
        assert fbank(wav, CFG).shape[0] == n_frames


def test_matches_torchaudio():
    """The model was trained on Kaldi features; a mismatch here silently degrades accuracy."""
    wav = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
    try:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        samples = torch.from_numpy(wav * 32767.0).unsqueeze(0)
    except Exception as exc:  # a torch/numpy ABI mismatch raises more than ImportError
        pytest.skip(f"torchaudio unavailable: {exc}")

    reference = kaldi.fbank(
        samples,
        num_mel_bins=CFG.n_mels,
        frame_length=CFG.frame_length_ms,
        frame_shift=CFG.frame_shift_ms,
        dither=0.0,
        sample_frequency=CFG.sample_rate,
        window_type=CFG.window,
        low_freq=CFG.low_freq,
        high_freq=CFG.high_freq,
        use_energy=False,
    ).numpy()

    ours = fbank(wav * 32767.0, CFG)
    assert ours.shape == reference.shape
    assert np.abs(ours - reference).max() < 1e-3
