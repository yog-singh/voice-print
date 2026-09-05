import numpy as np
import pytest

from voiceprint.exceptions import EnrollmentError
from voiceprint.store import SpeakerBook

DIM = 32


def _book():
    book = SpeakerBook("test_model", DIM)
    rng = np.random.default_rng(0)
    book.add("alice", rng.standard_normal((3, DIM)), {"source": "mic"})
    book.add("bob", rng.standard_normal((2, DIM)))
    return book


def test_add_and_lookup():
    book = _book()
    assert book.names == ["alice", "bob"]
    assert len(book) == 2
    assert "alice" in book
    assert book["alice"].n_utterances == 3
    assert book["alice"].metadata["source"] == "mic"


def test_centroid_is_unit_length():
    assert np.linalg.norm(_book()["alice"].centroid) == pytest.approx(1.0, abs=1e-6)


def test_adding_again_appends():
    book = _book()
    book.add("alice", np.random.randn(2, DIM))
    assert book["alice"].n_utterances == 5


def test_unknown_speaker_raises():
    with pytest.raises(EnrollmentError, match="not enrolled"):
        _book()["carol"]


def test_dimension_mismatch_raises():
    with pytest.raises(EnrollmentError, match="expected 32-d"):
        _book().add("dave", np.random.randn(1, 64))


def test_roundtrip(tmp_path):
    book = _book()
    path = book.save(tmp_path / "speakers")
    loaded = SpeakerBook.load(path, expect_model="test_model")

    assert loaded.names == book.names
    assert loaded["alice"].metadata == {"source": "mic"}
    np.testing.assert_allclose(loaded["alice"].centroid, book["alice"].centroid, atol=1e-6)


def test_loading_with_the_wrong_model_is_refused(tmp_path):
    path = _book().save(tmp_path / "speakers")
    with pytest.raises(EnrollmentError, match="built with model"):
        SpeakerBook.load(path, expect_model="other_model")


def test_all_embeddings_labels_line_up():
    embeddings, labels = _book().all_embeddings()
    assert embeddings.shape == (5, DIM)
    assert labels == ["alice"] * 3 + ["bob"] * 2


def test_empty_book_shapes():
    book = SpeakerBook("test_model", DIM)
    names, centroids = book.centroids()
    assert names == [] and centroids.shape == (0, DIM)
    embeddings, labels = book.all_embeddings()
    assert embeddings.shape == (0, DIM) and labels == []
