"""Tests for the vector index backends."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.vector_index import (
    ExactVectorIndex,
    angular_distance_to_cosine,
    annoy_available,
    create_index,
    load_index,
    resolve_backend,
    resolve_index_path,
)


class TestAngularConversion:
    def test_identical_vectors_score_one(self):
        assert angular_distance_to_cosine(0.0) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        # angular distance for cos=0 is sqrt(2)
        assert angular_distance_to_cosine(math.sqrt(2)) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_score_minus_one(self):
        assert angular_distance_to_cosine(2.0) == pytest.approx(-1.0)

    def test_result_is_clipped_to_valid_range(self):
        assert angular_distance_to_cosine(10.0) == -1.0


class TestExactVectorIndex:
    def test_query_returns_self_first(self, embeddings):
        index = ExactVectorIndex(dim=embeddings.shape[1])
        index.add_items(embeddings)
        indices, scores = index.query(embeddings[2], k=3)
        assert indices[0] == 2
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_scores_are_descending(self, exact_index, embeddings):
        _, scores = exact_index.query(embeddings[0], k=5)
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_cosine_similarities(self, exact_index, embeddings):
        indices, scores = exact_index.query(embeddings[0], k=5)
        for position, score in zip(indices, scores):
            expected = float(
                np.dot(embeddings[0], embeddings[position])
                / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[position]))
            )
            assert score == pytest.approx(expected, abs=1e-5)

    def test_k_is_clamped_to_index_size(self, exact_index):
        indices, _ = exact_index.query(np.ones(exact_index.dim), k=500)
        assert len(indices) == len(exact_index)

    def test_empty_index_returns_nothing(self):
        index = ExactVectorIndex(dim=4)
        assert index.query(np.ones(4), k=3) == ([], [])

    def test_dimension_mismatch_raises(self, exact_index):
        with pytest.raises(ValueError, match="dimension mismatch"):
            exact_index.query(np.ones(exact_index.dim + 1), k=1)

    def test_add_items_rejects_wrong_width(self):
        index = ExactVectorIndex(dim=4)
        with pytest.raises(ValueError, match="dimension mismatch"):
            index.add_items(np.zeros((3, 5)))

    def test_add_items_rejects_one_dimensional_input(self):
        index = ExactVectorIndex(dim=4)
        with pytest.raises(ValueError, match="2-D"):
            index.add_items(np.zeros(4))

    def test_zero_vectors_do_not_produce_nan(self):
        index = ExactVectorIndex(dim=4)
        index.add_items(np.zeros((2, 4)))
        _, scores = index.query(np.ones(4), k=2)
        assert not any(math.isnan(score) for score in scores)

    def test_add_items_appends(self, embeddings):
        index = ExactVectorIndex(dim=embeddings.shape[1])
        index.add_items(embeddings)
        index.add_items(embeddings)
        assert len(index) == 2 * len(embeddings)

    def test_save_and_load_roundtrip(self, embeddings, tmp_path):
        index = ExactVectorIndex(dim=embeddings.shape[1])
        index.add_items(embeddings)
        saved_to = index.save(tmp_path / "articles_index.annoy")
        assert saved_to.suffix == ".npz"

        reloaded = ExactVectorIndex.load(
            tmp_path / "articles_index.annoy", dim=embeddings.shape[1]
        )
        assert len(reloaded) == len(index)
        assert reloaded.query(embeddings[1], k=3) == index.query(embeddings[1], k=3)

    def test_load_rejects_dimension_mismatch(self, embeddings, tmp_path):
        index = ExactVectorIndex(dim=embeddings.shape[1])
        index.add_items(embeddings)
        index.save(tmp_path / "idx.annoy")
        with pytest.raises(ValueError, match="does not match"):
            ExactVectorIndex.load(tmp_path / "idx.annoy", dim=embeddings.shape[1] + 1)


class TestBackendSelection:
    def test_auto_resolves_to_an_installed_backend(self):
        assert resolve_backend("auto") in {"annoy", "exact"}

    def test_auto_prefers_annoy_when_available(self):
        expected = "annoy" if annoy_available() else "exact"
        assert resolve_backend("auto") == expected

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown index backend"):
            resolve_backend("faiss")

    def test_exact_backend_always_selectable(self):
        assert resolve_backend("exact") == "exact"

    def test_index_paths_differ_per_backend(self, tmp_path):
        base = tmp_path / "articles_index.annoy"
        assert resolve_index_path(base, "exact").suffix == ".npz"

    def test_create_index_honours_explicit_backend(self):
        assert create_index(dim=8, backend="exact").backend == "exact"

    def test_unsupported_metric_raises(self):
        with pytest.raises(ValueError, match="unsupported metric"):
            create_index(dim=8, backend="exact", metric="hamming")

    def test_non_positive_dim_raises(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            create_index(dim=0, backend="exact")

    def test_load_index_reports_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no index found"):
            load_index(tmp_path / "missing.annoy", dim=8, backend="auto")


@pytest.mark.skipif(not annoy_available(), reason="annoy is not installed")
class TestAnnoyBackend:
    """Only runs where the optional C++ extension could be installed."""

    def test_finds_the_query_vector_itself(self, embeddings):
        index = create_index(dim=embeddings.shape[1], backend="annoy", num_trees=20)
        index.add_items(embeddings)
        index.build()
        indices, scores = index.query(embeddings[2], k=3)
        assert indices[0] == 2
        assert scores[0] == pytest.approx(1.0, abs=1e-4)

    def test_scores_match_the_exact_backend(self, embeddings, exact_index):
        index = create_index(dim=embeddings.shape[1], backend="annoy", num_trees=50)
        index.add_items(embeddings)
        index.build()
        approx_ids, approx_scores = index.query(embeddings[0], k=3)
        exact_ids, exact_scores = exact_index.query(embeddings[0], k=3)
        assert approx_ids == exact_ids
        for approx, exact in zip(approx_scores, exact_scores):
            assert approx == pytest.approx(exact, abs=1e-4)

    def test_save_and_load_roundtrip(self, embeddings, tmp_path):
        index = create_index(dim=embeddings.shape[1], backend="annoy")
        index.add_items(embeddings)
        saved_to = index.save(tmp_path / "articles_index.annoy")
        assert saved_to.suffix == ".annoy"

        reloaded = load_index(
            tmp_path / "articles_index.annoy",
            dim=embeddings.shape[1],
            backend="annoy",
        )
        assert len(reloaded) == len(embeddings)
        assert reloaded.query(embeddings[1], k=2)[0] == index.query(embeddings[1], k=2)[0]

    def test_adding_after_build_raises(self, embeddings):
        index = create_index(dim=embeddings.shape[1], backend="annoy")
        index.add_items(embeddings)
        index.build()
        with pytest.raises(RuntimeError, match="already built"):
            index.add_items(embeddings)
