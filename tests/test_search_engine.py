"""Tests for the query-time search engine."""

from __future__ import annotations

import pickle

import pytest

from src.search_engine import (
    SearchEngine,
    load_metadata,
    load_search_engine,
    snippet,
)
from src.vector_index import ExactVectorIndex


class TestSnippet:
    def test_short_text_is_untouched(self):
        assert snippet("hello", 20) == "hello"

    def test_long_text_is_trimmed_with_ellipsis(self):
        assert snippet("abcdefghij", 4) == "abcd..."

    def test_zero_keeps_full_text(self):
        assert snippet("abcdefghij", 0) == "abcdefghij"

    def test_none_passes_through(self):
        assert snippet(None, 10) is None


class TestSearch:
    def test_returns_requested_number_of_results(self, engine):
        assert engine.search("climate", 2)["num_results"] == 2

    def test_ranks_topically_matching_article_first(self, engine):
        top = engine.search("neural networks training", 3)["results"][0]
        assert top["article_id"] == "article_00003"

    def test_scores_are_non_increasing(self, engine):
        scores = [hit["relevance_score"] for hit in engine.search("climate", 5)["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_result_carries_metadata(self, engine):
        hit = engine.search("markets rally", 1)["results"][0]
        assert hit["title"] == "markets rally quarterly earnings"
        assert hit["category"] == "Business"
        assert hit["source"] == "BBC"
        assert hit["published_date"] == "2024-03-01"

    def test_text_is_truncated_to_snippet_length(self, engine):
        hit = engine.search("markets rally", 1)["results"][0]
        assert hit["text"].endswith("...")
        assert len(hit["text"]) <= engine.snippet_chars + 3

    def test_echoes_the_stripped_query(self, engine):
        assert engine.search("  climate  ", 1)["query"] == "climate"

    def test_num_results_is_clamped_to_maximum(self, engine):
        assert engine.search("climate", 999)["num_results"] == engine.max_num_results

    def test_num_results_defaults_when_omitted(self, engine):
        assert engine.search("climate")["num_results"] == engine.default_num_results

    def test_zero_num_results_is_raised_to_one(self, engine):
        assert engine.search("climate", 0)["num_results"] == 1

    @pytest.mark.parametrize("bad_query", ["", "   ", None, 42])
    def test_invalid_queries_raise(self, engine, bad_query):
        with pytest.raises(ValueError):
            engine.search(bad_query, 1)

    def test_non_integer_num_results_raises(self, engine):
        with pytest.raises(ValueError, match="must be an integer"):
            engine.search("climate", "many")


class TestBatchSearch:
    def test_one_response_per_query(self, engine):
        assert len(engine.batch_search(["climate", "markets"], 2)) == 2

    def test_blank_queries_are_skipped(self, engine):
        assert len(engine.batch_search(["climate", "   ", ""], 2)) == 1


class TestEngineConstruction:
    def test_metadata_index_size_mismatch_raises(self, encoder, exact_index, metadata):
        metadata["num_articles"] = 99
        with pytest.raises(ValueError, match="rebuild the index"):
            SearchEngine(encoder=encoder, index=exact_index, metadata=metadata)

    def test_info_reports_backend_and_model(self, engine):
        info = engine.info()
        assert info["index_backend"] == "exact"
        assert info["embedding_model"] == "stub-hashing-encoder"
        assert info["num_articles"] == 5

    def test_missing_metadata_field_becomes_none(self, encoder, exact_index, metadata):
        metadata["sources"] = []
        engine = SearchEngine(encoder=encoder, index=exact_index, metadata=metadata)
        assert engine.search("climate", 1)["results"][0]["source"] is None


class TestLoading:
    def test_load_metadata_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata not found"):
            load_metadata(tmp_path / "nope.pkl")

    def test_load_metadata_rejects_foreign_pickle(self, tmp_path):
        path = tmp_path / "meta.pkl"
        path.write_bytes(pickle.dumps({"unrelated": True}))
        with pytest.raises(ValueError, match="does not look like"):
            load_metadata(path)

    def test_load_search_engine_from_disk(self, tmp_path, embeddings, metadata, encoder):
        index = ExactVectorIndex(dim=embeddings.shape[1])
        index.add_items(embeddings)
        index.save(tmp_path / "articles_index.annoy")
        (tmp_path / "metadata.pkl").write_bytes(pickle.dumps(metadata))

        engine = load_search_engine(
            index_path=tmp_path / "articles_index.annoy",
            metadata_path=tmp_path / "metadata.pkl",
            backend="exact",
            encoder=encoder,
        )
        assert engine.num_articles == 5
        assert engine.search("climate warming", 1)["results"][0]["article_id"] == (
            "article_00001"
        )
