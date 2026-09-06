import numpy as np
import pytest

from voicefingerprint.audio import load_audio, normalize_volume, preprocess, resample
from voicefingerprint.config import PreprocessConfig
from voicefingerprint.exceptions import AudioError
from voicefingerprint.vad import EnergyVAD, apply_vad, mask_to_segments

SR = 16000


def test_load_from_array_requires_sample_rate():
    with pytest.raises(AudioError, match="source_sr is required"):
        load_audio(np.zeros(100, dtype=np.float32), SR)


def test_load_downmixes_stereo():
    stereo = np.stack([np.ones(1000), -np.ones(1000)], axis=1).astype(np.float32)
    assert load_audio(stereo, SR, source_sr=SR).shape == (1000,)


def test_missing_file_raises():
    with pytest.raises(AudioError, match="not found"):
        load_audio("/nope/missing.wav", SR)


def test_resample_changes_length_proportionally():
    wav = np.random.randn(SR).astype(np.float32)
    assert abs(len(resample(wav, SR, 8000)) - 8000) <= 2


def test_normalize_volume_hits_the_target():
    wav = 0.001 * np.random.randn(SR).astype(np.float32)
    out = normalize_volume(wav, -26.0)
    assert 20 * np.log10(np.sqrt(np.mean(out**2))) == pytest.approx(-26.0, abs=0.1)


def test_normalize_volume_leaves_silence_alone():
    silence = np.zeros(SR, dtype=np.float32)
    np.testing.assert_array_equal(normalize_volume(silence, -26.0), silence)


def test_energy_vad_drops_a_silent_gap(wav_a):
    signal = np.concatenate([wav_a[: SR], np.zeros(2 * SR, dtype=np.float32), wav_a[: SR]])
    trimmed = apply_vad(signal, EnergyVAD(), SR)
    assert len(trimmed) < len(signal)
    assert len(trimmed) > 1.5 * SR


def test_vad_falls_back_when_it_would_remove_everything():
    silence = np.zeros(3 * SR, dtype=np.float32)
    assert len(apply_vad(silence, EnergyVAD(), SR)) == len(silence)


def test_mask_to_segments():
    mask = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert mask_to_segments(mask) == [(1, 3), (5, 6)]
    assert mask_to_segments(np.ones(4, dtype=bool)) == [(0, 4)]


def test_preprocess_end_to_end(wav_a):
    out = preprocess(wav_a, PreprocessConfig(vad_backend="energy"), source_sr=SR)
    assert out.dtype == np.float32 and out.ndim == 1 and len(out) > 0


def test_silero_separates_speech_from_silence(wav_a):
    """Skipped when the Silero weights cannot be fetched."""
    pytest.importorskip("onnxruntime")
    from voicefingerprint.exceptions import ModelError
    from voicefingerprint.vad import SileroVAD

    try:
        vad = SileroVAD()
    except ModelError as exc:
        pytest.skip(f"Silero unavailable: {exc}")

    signal = np.concatenate([wav_a[: 2 * SR], np.zeros(2 * SR, dtype=np.float32), wav_a[2 * SR :]])
    mask = vad.speech_mask(signal, SR)
    assert mask[2 * SR : 4 * SR].mean() < 0.5
    assert len(mask) == len(signal)


def test_silero_rejects_unsupported_sample_rates():
    from voicefingerprint.exceptions import ModelError
    from voicefingerprint.vad import SileroVAD

    try:
        vad = SileroVAD()
    except ModelError as exc:
        pytest.skip(f"Silero unavailable: {exc}")
    with pytest.raises(ModelError, match="8 or 16 kHz"):
        vad.speech_mask(np.zeros(44100, dtype=np.float32), 44100)
