"""End-to-end tests for the embedding and indexing pipeline steps.

The SBERT model is replaced with the stub encoder so the whole pipeline runs
offline in milliseconds.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.build_annoy_index import build_annoy_index, build_index_from_embeddings
from src.sbert_embeddings import (
    METADATA_TEXT_CHARS,
    build_metadata,
    generate_embeddings,
    texts_to_embed,
)
from src.search_engine import load_search_engine


class StubEmbedder:
    """Adapts the hashing encoder to the SBERTEmbedder interface."""

    def __init__(self, encoder):
        self._encoder = encoder
        self.model_name = encoder.model_name
        self.embedding_dim = encoder.dim

    def encode(self, texts, batch_size: int = 32, show_progress: bool = False):
        return self._encoder.encode(texts)


@pytest.fixture
def processed_csv(tmp_path, corpus):
    df = pd.DataFrame(corpus)
    df["processed_text"] = df["title"] + " " + df["text"]
    path = tmp_path / "processed.csv"
    df.to_csv(path, index=False)
    return path


class TestTextsToEmbed:
    def test_prefers_the_processed_column(self, corpus):
        df = pd.DataFrame(corpus)
        df["processed_text"] = "lemmatised"
        assert texts_to_embed(df) == ["lemmatised"] * len(corpus)

    def test_falls_back_to_title_and_text(self, corpus):
        df = pd.DataFrame(corpus)
        df["processed_text"] = ""
        assert texts_to_embed(df)[0].startswith("climate warming oceans")

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="neither"):
            texts_to_embed(pd.DataFrame({"other": [1, 2]}))


class TestBuildMetadata:
    def test_captures_every_article(self, corpus):
        metadata = build_metadata(pd.DataFrame(corpus), "stub", 8)
        assert metadata["num_articles"] == len(corpus)
        assert metadata["embedding_dimension"] == 8
        assert metadata["article_ids"][0] == "article_00001"

    def test_absent_columns_become_empty_lists(self, corpus):
        df = pd.DataFrame(corpus).drop(columns=["subcategory"])
        assert build_metadata(df, "stub", 8)["subcategories"] == []

    def test_bodies_are_truncated(self, corpus):
        df = pd.DataFrame(corpus)
        df["text"] = "x" * (METADATA_TEXT_CHARS * 2)
        metadata = build_metadata(df, "stub", 8)
        assert len(metadata["texts"][0]) == METADATA_TEXT_CHARS

    def test_nan_bodies_survive(self, corpus):
        df = pd.DataFrame(corpus)
        df.loc[0, "text"] = np.nan
        assert build_metadata(df, "stub", 8)["texts"][0] is None


class TestGenerateEmbeddings:
    def test_writes_embeddings_and_metadata(self, tmp_path, processed_csv, encoder):
        embeddings, metadata = generate_embeddings(
            processed_csv,
            tmp_path / "embeddings.npy",
            tmp_path / "metadata.pkl",
            embedder=StubEmbedder(encoder),
        )
        assert embeddings.shape == (5, encoder.dim)
        assert (tmp_path / "embeddings.npy").exists()
        assert metadata["embedding_model"] == "stub-hashing-encoder"
        with open(tmp_path / "metadata.pkl", "rb") as handle:
            assert pickle.load(handle)["num_articles"] == 5

    def test_missing_input_raises(self, tmp_path, encoder):
        with pytest.raises(FileNotFoundError, match="processed data not found"):
            generate_embeddings(
                tmp_path / "missing.csv",
                tmp_path / "e.npy",
                tmp_path / "m.pkl",
                embedder=StubEmbedder(encoder),
            )

    def test_empty_input_raises(self, tmp_path, encoder):
        path = tmp_path / "empty.csv"
        pd.DataFrame({"title": [], "text": [], "processed_text": []}).to_csv(
            path, index=False
        )
        with pytest.raises(ValueError, match="contains no rows"):
            generate_embeddings(
                path,
                tmp_path / "e.npy",
                tmp_path / "m.pkl",
                embedder=StubEmbedder(encoder),
            )


class TestBuildIndex:
    def test_index_holds_every_vector(self, embeddings):
        index = build_index_from_embeddings(embeddings, backend="exact")
        assert len(index) == len(embeddings)
        assert index.dim == embeddings.shape[1]

    def test_empty_embeddings_raise(self):
        with pytest.raises(ValueError, match="non-empty 2-D"):
            build_index_from_embeddings(np.zeros((0, 4)), backend="exact")

    def test_missing_embeddings_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="embeddings not found"):
            build_annoy_index(
                tmp_path / "missing.npy",
                tmp_path / "metadata.pkl",
                tmp_path / "index.annoy",
                backend="exact",
            )

    def test_row_count_mismatch_is_caught(self, tmp_path, embeddings, metadata):
        np.save(tmp_path / "embeddings.npy", embeddings[:3])
        (tmp_path / "metadata.pkl").write_bytes(pickle.dumps(metadata))
        with pytest.raises(ValueError, match="row mismatch"):
            build_annoy_index(
                tmp_path / "embeddings.npy",
                tmp_path / "metadata.pkl",
                tmp_path / "index.annoy",
                backend="exact",
            )

    def test_dimension_mismatch_is_caught(self, tmp_path, embeddings, metadata):
        np.save(tmp_path / "embeddings.npy", embeddings)
        metadata["embedding_dimension"] = embeddings.shape[1] + 1
        (tmp_path / "metadata.pkl").write_bytes(pickle.dumps(metadata))
        with pytest.raises(ValueError, match="dimension mismatch"):
            build_annoy_index(
                tmp_path / "embeddings.npy",
                tmp_path / "metadata.pkl",
                tmp_path / "index.annoy",
                backend="exact",
            )


class TestFullPipeline:
    def test_csv_to_searchable_engine(self, tmp_path, processed_csv, encoder):
        generate_embeddings(
            processed_csv,
            tmp_path / "embeddings.npy",
            tmp_path / "metadata.pkl",
            embedder=StubEmbedder(encoder),
        )
        build_annoy_index(
            tmp_path / "embeddings.npy",
            tmp_path / "metadata.pkl",
            tmp_path / "articles_index.annoy",
            backend="exact",
        )
        engine = load_search_engine(
            index_path=tmp_path / "articles_index.annoy",
            metadata_path=tmp_path / "metadata.pkl",
            backend="exact",
            encoder=encoder,
        )
        results = engine.search("compiler optimisation release", 2)["results"]
        assert results[0]["article_id"] == "article_00004"
