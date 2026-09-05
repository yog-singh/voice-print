import numpy as np
import pytest

from voiceprint.scoring import ASNorm, calibrate_threshold, cosine, cosine_matrix, equal_error_rate, l2_normalize


def test_cosine_bounds():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, -v) == pytest.approx(-1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_is_scale_invariant():
    a, b = np.random.randn(64), np.random.randn(64)
    assert cosine(a, b) == pytest.approx(cosine(3.7 * a, 0.2 * b))


def test_l2_normalize_handles_zero_vectors():
    out = l2_normalize(np.zeros((2, 8)))
    assert np.isfinite(out).all()


def test_cosine_matrix_shape():
    assert cosine_matrix(np.random.randn(3, 16), np.random.randn(5, 16)).shape == (3, 5)


def test_eer_on_perfectly_separable_scores():
    scores = [0.9, 0.85, 0.8, 0.2, 0.1, 0.05]
    labels = [1, 1, 1, 0, 0, 0]
    eer, threshold = equal_error_rate(scores, labels)
    assert eer == pytest.approx(0.0)
    assert 0.2 < threshold <= 0.8


def test_eer_on_random_scores_is_near_half():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000)
    eer, _ = equal_error_rate(scores, labels)
    assert 0.4 < eer < 0.6


def test_eer_requires_both_classes():
    with pytest.raises(ValueError):
        equal_error_rate([0.1, 0.2], [1, 1])


def test_calibrate_recovers_a_separating_threshold():
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((4, 32))
    embeddings, ids = [], []
    for i, center in enumerate(centers):
        for _ in range(5):
            embeddings.append(center + 0.05 * rng.standard_normal(32))
            ids.append(f"spk{i}")

    report = calibrate_threshold(np.array(embeddings), ids)
    assert report["n_speakers"] == 4
    assert report["eer"] < 0.05
    assert -1 <= report["threshold"] <= 1


def test_asnorm_preserves_ranking_but_rescales():
    rng = np.random.default_rng(2)
    cohort = rng.standard_normal((50, 32))
    a = rng.standard_normal(32)
    close, far = a + 0.01 * rng.standard_normal(32), rng.standard_normal(32)

    norm = ASNorm(cohort, top_k=20)
    assert norm(a, close) > norm(a, far)
    assert norm(a, close) != pytest.approx(cosine(a, close))
