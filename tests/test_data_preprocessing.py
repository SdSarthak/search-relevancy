"""Tests for text cleaning and the preprocessing pipeline.

The cleaning tests need no models. The lemmatisation tests are skipped when
the spaCy model is not installed, so the suite still runs on a bare checkout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    OUTPUT_COLUMNS,
    NewsArticlePreprocessor,
    preprocess_dataframe,
    preprocess_dataset,
)


def spacy_model_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        return False
    return True


requires_spacy = pytest.mark.skipif(
    not spacy_model_available(), reason="spaCy model en_core_web_sm not installed"
)


@pytest.fixture
def preprocessor() -> NewsArticlePreprocessor:
    return NewsArticlePreprocessor()


class TestCleanText:
    def test_lowercases_by_default(self, preprocessor):
        assert preprocessor.clean_text("Climate CHANGE") == "climate change"

    def test_keeps_case_when_disabled(self):
        cleaner = NewsArticlePreprocessor(lowercase=False)
        assert cleaner.clean_text("Climate CHANGE") == "Climate CHANGE"

    @pytest.mark.parametrize(
        "raw",
        ["visit http://example.com/story now", "visit https://example.com now",
         "visit www.example.com now"],
    )
    def test_strips_urls(self, preprocessor, raw):
        assert "example.com" not in preprocessor.clean_text(raw)

    def test_strips_emails(self, preprocessor):
        assert preprocessor.clean_text("mail me@example.com now") == "mail now"

    def test_strips_html_tags(self, preprocessor):
        assert preprocessor.clean_text("<p>hello <b>world</b></p>") == "hello world"

    def test_collapses_whitespace(self, preprocessor):
        assert preprocessor.clean_text("a\n\n  b\tc  ") == "a b c"

    def test_keeps_digits_by_default(self, preprocessor):
        assert "2024" in preprocessor.clean_text("Report 2024 results")

    def test_removes_non_letters_when_configured(self):
        cleaner = NewsArticlePreprocessor(remove_punctuation=True)
        assert cleaner.clean_text("Report 2024: results!") == "report results"

    @pytest.mark.parametrize("empty", [None, float("nan"), np.nan, ""])
    def test_missing_values_become_empty_strings(self, preprocessor, empty):
        assert preprocessor.clean_text(empty) == ""

    def test_non_string_input_is_coerced(self, preprocessor):
        assert preprocessor.clean_text(12345) == "12345"


class TestPreprocess:
    def test_text_below_minimum_length_is_dropped(self, preprocessor):
        assert preprocessor.preprocess("hi") == ""

    @requires_spacy
    def test_lemmatises_and_drops_stopwords(self, preprocessor):
        result = preprocessor.preprocess("The oceans are rising quickly this year")
        assert "the" not in result.split()
        assert "ocean" in result

    @requires_spacy
    def test_keeps_stopwords_when_configured(self):
        cleaner = NewsArticlePreprocessor(remove_stopwords=False)
        assert "the" in cleaner.preprocess("The oceans are rising quickly").split()

    @requires_spacy
    def test_batch_matches_single_document_path(self, preprocessor):
        texts = ["The oceans are rising quickly", "Markets rallied on strong earnings"]
        assert preprocessor.preprocess_many(texts) == [
            preprocessor.preprocess(text) for text in texts
        ]

    def test_batch_preserves_positions_for_short_rows(self, preprocessor):
        results = preprocessor.preprocess_many(["hi", ""])
        assert results == ["", ""]


class TestPreprocessDataframe:
    @requires_spacy
    def test_produces_the_expected_columns(self, corpus):
        result = preprocess_dataframe(pd.DataFrame(corpus))
        assert list(result.columns) == list(OUTPUT_COLUMNS)
        assert len(result) == len(corpus)
        assert result["processed_text"].str.len().gt(0).all()

    @requires_spacy
    def test_backfills_optional_columns(self, corpus):
        df = pd.DataFrame(corpus).drop(columns=["source", "subcategory"])
        result = preprocess_dataframe(df)
        assert (result["source"] == "").all()

    @requires_spacy
    def test_generates_article_ids_when_absent(self, corpus):
        df = pd.DataFrame(corpus).drop(columns=["article_id"])
        result = preprocess_dataframe(df)
        assert result["article_id"].iloc[0] == "article_00001"

    @requires_spacy
    def test_rows_that_reduce_to_nothing_are_dropped(self, corpus):
        df = pd.DataFrame(corpus + [{"article_id": "x", "title": "", "text": ""}])
        assert len(preprocess_dataframe(df)) == len(corpus)

    def test_missing_required_column_raises(self, corpus):
        df = pd.DataFrame(corpus).drop(columns=["text"])
        with pytest.raises(ValueError, match="missing required column"):
            preprocess_dataframe(df)


class TestPreprocessDataset:
    def test_missing_input_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="raw data not found"):
            preprocess_dataset(tmp_path / "missing.csv", tmp_path / "out.csv")

    @requires_spacy
    def test_writes_a_processed_csv(self, tmp_path, corpus):
        raw = tmp_path / "raw.csv"
        out = tmp_path / "processed" / "out.csv"
        pd.DataFrame(corpus).to_csv(raw, index=False)

        result = preprocess_dataset(raw, out)
        assert out.exists()
        assert len(pd.read_csv(out)) == len(result) == len(corpus)

    @requires_spacy
    def test_sampling_is_deterministic(self, tmp_path, corpus):
        raw = tmp_path / "raw.csv"
        pd.DataFrame(corpus).to_csv(raw, index=False)
        first = preprocess_dataset(raw, tmp_path / "a.csv", sample_size=3)
        second = preprocess_dataset(raw, tmp_path / "b.csv", sample_size=3)
        assert first["article_id"].tolist() == second["article_id"].tolist()
        assert len(first) == 3
