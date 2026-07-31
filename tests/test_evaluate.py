"""Tests for the relevancy metrics and the evaluation harness."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import (
    build_known_item_queries,
    evaluate_approximation,
    evaluate_ranking,
    format_report,
    load_query_set,
    ndcg_at_k,
    overlap_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestMetrics:
    def test_precision_counts_hits_in_top_k(self):
        assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == pytest.approx(0.5)

    def test_precision_ignores_hits_below_the_cut_off(self):
        assert precision_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_recall_is_over_the_relevant_set(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == pytest.approx(0.5)

    def test_recall_saturates_at_one(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(1.0)

    def test_empty_relevant_set_scores_zero(self):
        assert recall_at_k(["a"], [], 1) == 0.0
        assert precision_at_k(["a"], [], 1) == 0.0
        assert ndcg_at_k(["a"], [], 1) == 0.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be positive"):
            precision_at_k(["a"], {"a"}, k)

    def test_reciprocal_rank_uses_first_hit(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_reciprocal_rank_is_zero_without_a_hit(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_ndcg_is_one_for_a_perfect_ranking(self):
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_ndcg_penalises_late_hits(self):
        good = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        bad = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert 0 < bad < good == pytest.approx(1.0)

    def test_overlap_is_one_for_identical_rankings(self):
        assert overlap_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == pytest.approx(1.0)

    def test_overlap_ignores_order(self):
        assert overlap_at_k(["c", "b", "a"], ["a", "b", "c"], 3) == pytest.approx(1.0)

    def test_overlap_counts_missing_neighbours(self):
        assert overlap_at_k(["a", "z"], ["a", "b"], 2) == pytest.approx(0.5)

    def test_overlap_of_empty_exact_list(self):
        assert overlap_at_k(["a"], [], 3) == 0.0


class TestQuerySets:
    def test_known_item_queries_cover_the_corpus(self, metadata):
        queries = build_known_item_queries(metadata, sample_size=None)
        assert len(queries) == metadata["num_articles"]
        assert queries[0]["relevant_ids"] == ["article_00001"]
        assert queries[0]["category"] == "Science"

    def test_sampling_is_deterministic(self, metadata):
        first = build_known_item_queries(metadata, sample_size=3, seed=7)
        second = build_known_item_queries(metadata, sample_size=3, seed=7)
        assert [q["query"] for q in first] == [q["query"] for q in second]
        assert len(first) == 3

    def test_load_query_set_from_json(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(
            json.dumps([{"query": "climate", "relevant_ids": ["article_00001"]}]),
            encoding="utf-8",
        )
        queries = load_query_set(path)
        assert queries[0]["query"] == "climate"
        assert queries[0]["relevant_ids"] == ["article_00001"]

    def test_load_query_set_accepts_a_scalar_relevant_id(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(
            json.dumps({"queries": [{"query": "climate", "relevant_ids": "a1"}]}),
            encoding="utf-8",
        )
        assert load_query_set(path)[0]["relevant_ids"] == ["a1"]

    def test_load_query_set_rejects_malformed_entries(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(json.dumps([{"no_query": 1}]), encoding="utf-8")
        with pytest.raises(ValueError, match="malformed query entry"):
            load_query_set(path)

    def test_load_query_set_rejects_empty_file(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty list"):
            load_query_set(path)


class TestEvaluateRanking:
    def test_known_item_retrieval_is_perfect_on_the_toy_corpus(self, engine, metadata):
        queries = build_known_item_queries(metadata, sample_size=None)
        report = evaluate_ranking(engine, queries, k_values=[1, 3])
        assert report["num_queries"] == 5
        assert report["metrics"]["@1"]["recall"] == pytest.approx(1.0)
        assert report["mrr"] == pytest.approx(1.0)

    def test_precision_at_k_falls_with_one_relevant_document(self, engine, metadata):
        queries = build_known_item_queries(metadata, sample_size=None)
        report = evaluate_ranking(engine, queries, k_values=[1, 5])
        assert report["metrics"]["@5"]["precision"] == pytest.approx(0.2)

    def test_unrelated_labels_score_zero(self, engine):
        queries = [{"query": "climate", "relevant_ids": ["not_in_corpus"]}]
        report = evaluate_ranking(engine, queries, k_values=[3])
        assert report["metrics"]["@3"]["recall"] == 0.0
        assert report["mrr"] == 0.0

    def test_category_match_is_reported_when_labelled(self, engine, metadata):
        queries = build_known_item_queries(metadata, sample_size=None)
        report = evaluate_ranking(engine, queries, k_values=[3])
        assert report["top1_category_match"] == pytest.approx(1.0)

    def test_latency_is_recorded(self, engine, metadata):
        report = evaluate_ranking(
            engine, build_known_item_queries(metadata, sample_size=None), [1]
        )
        assert report["latency_ms"]["mean"] >= 0.0
        assert "p95" in report["latency_ms"]

    def test_empty_query_set_raises(self, engine):
        with pytest.raises(ValueError, match="query set is empty"):
            evaluate_ranking(engine, [], [1])

    def test_invalid_k_values_raise(self, engine, metadata):
        queries = build_known_item_queries(metadata, sample_size=None)
        with pytest.raises(ValueError, match="at least one positive integer"):
            evaluate_ranking(engine, queries, k_values=[0, -3])


class TestEvaluateApproximation:
    def test_exact_backend_matches_itself(self, exact_index, embeddings):
        report = evaluate_approximation(exact_index, embeddings, embeddings, k=3)
        assert report["backend"] == "exact"
        assert report["recall_vs_exact@3"] == pytest.approx(1.0)
        assert report["num_queries"] == len(embeddings)

    def test_latency_fields_present(self, exact_index, embeddings):
        report = evaluate_approximation(exact_index, embeddings, embeddings[:2], k=2)
        assert report["latency_ms"]["approximate_mean"] >= 0.0
        assert report["latency_ms"]["exact_mean"] >= 0.0


class TestFormatReport:
    def test_renders_both_sections(self, engine, metadata, exact_index, embeddings):
        report = {
            "ranking": evaluate_ranking(
                engine, build_known_item_queries(metadata, sample_size=None), [1, 3]
            ),
            "approximation": evaluate_approximation(
                exact_index, embeddings, np.asarray(embeddings[:2]), k=3
            ),
        }
        text = format_report(report)
        assert "Search relevancy report" in text
        assert "MRR" in text
        assert "recall vs exact@3" in text

    def test_handles_a_partial_report(self):
        assert "Search relevancy report" in format_report({})
