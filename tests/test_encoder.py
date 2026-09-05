"""Live-model tests. Skipped automatically when the ONNX weights cannot be fetched."""

import numpy as np
import pytest

from voiceprint import Config, VoiceRecognizer
from voiceprint.exceptions import EnrollmentError, ModelError

SR = 16000


def test_signature_matches_the_registry(encoder):
    info = encoder.describe()
    assert info["embedding_dim"] == 256
    assert info["sample_rate"] == SR


def test_embeddings_are_unit_length(encoder, wav_a):
    embedding = encoder.embed_utterance(wav_a)
    assert embedding.shape == (256,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic(encoder, wav_a):
    np.testing.assert_allclose(encoder.embed_utterance(wav_a), encoder.embed_utterance(wav_a), atol=1e-6)


def test_same_source_scores_above_different_source(encoder, wav_a, wav_b):
    from voiceprint.scoring import cosine

    first, second = encoder.embed_utterance(wav_a[: 2 * SR]), encoder.embed_utterance(wav_a[2 * SR :])
    other = encoder.embed_utterance(wav_b)
    assert cosine(first, second) > cosine(first, other)


def test_window_count_tracks_the_rate(encoder, wav_a):
    sparse, _ = encoder.embed_timeline(wav_a, rate=1.0)
    dense, slices = encoder.embed_timeline(wav_a, rate=8.0)
    assert len(dense) > len(sparse)
    assert len(dense) == len(slices)


def test_audio_shorter_than_one_frame_raises(encoder):
    with pytest.raises(ModelError, match="need at least"):
        encoder.embed_utterance(np.zeros(50, dtype=np.float32))


def test_recognizer_enroll_identify_roundtrip(encoder, tmp_path, wav_a, wav_b):
    recognizer = VoiceRecognizer(Config())
    recognizer.enroll("a", [wav_a[: 2 * SR], wav_a[2 * SR :]], source_sr=SR)
    recognizer.enroll("b", [wav_b[: 2 * SR], wav_b[2 * SR :]], source_sr=SR)

    matches = recognizer.identify(wav_a[: 3 * SR], source_sr=SR)
    assert matches[0].name == "a"

    path = recognizer.save(tmp_path / "book")
    recognizer.load(path)
    assert recognizer.book.names == ["a", "b"]

    report = recognizer.calibrate()
    assert report["n_speakers"] == 2
    assert recognizer.threshold == report["threshold"]


def test_identify_without_enrollment_raises(encoder, wav_a):
    with pytest.raises(EnrollmentError, match="no speakers enrolled"):
        VoiceRecognizer(Config()).identify(wav_a, source_sr=SR)
