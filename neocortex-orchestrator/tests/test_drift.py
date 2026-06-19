import numpy as np

from neocortex.drift.semantic_drift import SemanticDriftDetector
from neocortex.embeddings import EmbeddingProvider, cosine_similarity


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) > 0.999


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector_is_safe():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert cosine_similarity(a, b) == 0.0


def test_drift_score_no_reference_is_neutral():
    detector = SemanticDriftDetector(EmbeddingProvider())
    assert detector.drift_score("anything", []) == 0.5


def test_drift_score_low_when_response_matches_reference():
    detector = SemanticDriftDetector(EmbeddingProvider())
    text = "The Eiffel Tower was completed in 1889 and is located in Paris."
    score = detector.drift_score(text, [text])
    assert score < 0.05


def test_drift_score_high_when_response_is_unrelated():
    detector = SemanticDriftDetector(EmbeddingProvider())
    response = "The stock market rallied sharply on Tuesday amid earnings optimism."
    reference = ["The mitochondria is the powerhouse of the eukaryotic cell."]
    score = detector.drift_score(response, reference)
    assert score > 0.5
