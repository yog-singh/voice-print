import numpy as np
import pytest

SR = 16000


def _voice_like(seconds: float, f0: float, seed: int) -> np.ndarray:
    """A crude harmonic-plus-noise signal. Not speech, but stable and non-degenerate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    wav = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, np.pi)) for h in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)
    return (0.1 * envelope * wav + 0.005 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="session")
def sample_rate():
    return SR


@pytest.fixture
def wav_a():
    return _voice_like(4.0, 110.0, seed=1)


@pytest.fixture
def wav_b():
    return _voice_like(4.0, 210.0, seed=2)


@pytest.fixture(scope="session")
def encoder():
    from voicefingerprint import SpeakerEncoder
    from voicefingerprint.exceptions import ModelError

    try:
        return SpeakerEncoder()
    except ModelError as exc:
        pytest.skip(f"encoder unavailable: {exc}")
