import pytest

from transcribe.speakers import (
    _cosine,
    load_embeddings,
    match_speakers,
    save_embeddings,
)


def test_cosine_identical():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "speakers.json"
    embs = {"Dave": [1.0, 0.0, 0.0], "Nastassia": [0.0, 1.0, 0.0]}
    save_embeddings(embs, path)
    loaded = load_embeddings(path)
    assert loaded["Dave"] == pytest.approx([1.0, 0.0, 0.0])
    assert loaded["Nastassia"] == pytest.approx([0.0, 1.0, 0.0])


def test_save_merges_with_existing(tmp_path):
    path = tmp_path / "speakers.json"
    save_embeddings({"Dave": [1.0, 0.0]}, path)
    save_embeddings({"Nastassia": [0.0, 1.0]}, path)
    loaded = load_embeddings(path)
    assert "Dave" in loaded
    assert "Nastassia" in loaded


def test_save_overwrites_same_name(tmp_path):
    path = tmp_path / "speakers.json"
    save_embeddings({"Dave": [1.0, 0.0]}, path)
    save_embeddings({"Dave": [0.5, 0.5]}, path)
    loaded = load_embeddings(path)
    assert loaded["Dave"] == pytest.approx([0.5, 0.5])


def test_load_missing_returns_empty(tmp_path):
    assert load_embeddings(tmp_path / "nonexistent.json") == {}


def test_match_speakers_exact(tmp_path):
    cluster_embs = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}
    known = {"Dave": [1.0, 0.0], "Nastassia": [0.0, 1.0]}
    mapping = match_speakers(cluster_embs, known, threshold=0.9)
    assert mapping["SPEAKER_00"] == "Dave"
    assert mapping["SPEAKER_01"] == "Nastassia"


def test_match_speakers_below_threshold_keeps_label():
    cluster_embs = {"SPEAKER_00": [1.0, 0.0]}
    known = {"Dave": [0.0, 1.0]}  # orthogonal → similarity 0.0
    mapping = match_speakers(cluster_embs, known, threshold=0.75)
    assert mapping["SPEAKER_00"] == "SPEAKER_00"


def test_match_speakers_empty_known():
    cluster_embs = {"SPEAKER_00": [1.0, 0.0]}
    mapping = match_speakers(cluster_embs, {})
    assert mapping["SPEAKER_00"] == "SPEAKER_00"
