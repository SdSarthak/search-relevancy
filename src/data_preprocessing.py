"""Clean and normalise raw news articles before embedding them.

The spaCy pipeline is loaded lazily so that importing this module (for tests,
or just to reuse :func:`clean_text`) never triggers a model download.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Allow both "python -m src.data_preprocessing" and direct script execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Columns carried through to the processed dataset. Only ``title``/``text``
# are required; the rest are filled with empty values when absent.
REQUIRED_COLUMNS = ("title", "text")
OPTIONAL_COLUMNS = (
    "article_id",
    "category",
    "subcategory",
    "published_date",
    "source",
)
OUTPUT_COLUMNS = (
    "article_id",
    "category",
    "subcategory",
    "title",
    "published_date",
    "source",
    "text",
    "processed_text",
)

_URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
_HTML_RE = re.compile(r"<[^>]*>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")

_SPACY_PIPELINE = None


def load_spacy(model_name: str = "en_core_web_sm"):
    """Load (and cache) the spaCy pipeline used for lemmatisation.

    Raises a clear, actionable error instead of silently shelling out to
    ``spacy download`` at import time.
    """
    global _SPACY_PIPELINE
    if _SPACY_PIPELINE is not None:
        return _SPACY_PIPELINE

    import spacy

    try:
        # The tagger/parser/NER are not needed for lemmatisation.
        _SPACY_PIPELINE = spacy.load(model_name, disable=["parser", "ner"])
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc
    return _SPACY_PIPELINE


class NewsArticlePreprocessor:
    """Preprocess news article text for SBERT embedding."""

    #: spaCy refuses documents longer than this, and long bodies add nothing.
    MAX_CHARS = 100_000

    def __init__(
        self,
        remove_stopwords: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        min_text_length: int = 10,
        spacy_model: str = "en_core_web_sm",
    ):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.min_text_length = min_text_length
        self.spacy_model = spacy_model

    def clean_text(self, text) -> str:
        """Strip URLs, emails and markup, then squash whitespace."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        text = str(text)
        if self.lowercase:
            text = text.lower()

        text = _URL_RE.sub(" ", text)
        text = _EMAIL_RE.sub(" ", text)
        text = _HTML_RE.sub(" ", text)
        if self.remove_punctuation:
            text = _NON_ALPHA_RE.sub(" ", text)

        return " ".join(text.split())

    def tokenize_and_lemmatize(self, text: str) -> str:
        """Lemmatise ``text``, dropping stopwords, punctuation and whitespace."""
        if not text:
            return ""

        nlp = load_spacy(self.spacy_model)
        doc = nlp(text[: self.MAX_CHARS])

        tokens: List[str] = []
        for token in doc:
            if self.remove_stopwords and token.is_stop:
                continue
            if token.is_punct or token.is_space:
                continue
            lemma = token.lemma_.strip()
            if lemma:
                tokens.append(lemma)
        return " ".join(tokens)

    def preprocess(self, text) -> str:
        """Full pipeline: clean, then lemmatise."""
        cleaned = self.clean_text(text)
        if len(cleaned) < self.min_text_length:
            return ""
        return self.tokenize_and_lemmatize(cleaned)

    def preprocess_many(self, texts, batch_size: int = 64) -> List[str]:
        """Vectorised variant of :meth:`preprocess` using ``nlp.pipe``.

        Roughly an order of magnitude faster than calling :meth:`preprocess`
        per row on large datasets.
        """
        cleaned = [self.clean_text(text) for text in texts]
        keep = [i for i, text in enumerate(cleaned) if len(text) >= self.min_text_length]
        results = [""] * len(cleaned)
        if not keep:
            return results

        nlp = load_spacy(self.spacy_model)
        docs = nlp.pipe(
            (cleaned[i][: self.MAX_CHARS] for i in keep), batch_size=batch_size
        )
        for position, doc in zip(keep, docs):
            tokens = [
                token.lemma_.strip()
                for token in doc
                if not (self.remove_stopwords and token.is_stop)
                and not token.is_punct
                and not token.is_space
                and token.lemma_.strip()
            ]
            results[position] = " ".join(tokens)
        return results


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and backfill the optional ones."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"input data is missing required column(s): {', '.join(missing)}. "
            f"Expected at least {', '.join(REQUIRED_COLUMNS)}."
        )

    df = df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            logger.warning("column '%s' missing from input; filling with blanks", column)
            df[column] = ""
    if (df["article_id"] == "").all():
        df["article_id"] = [f"article_{i + 1:05d}" for i in range(len(df))]
    return df


def preprocess_dataframe(
    df: pd.DataFrame, preprocessor: Optional[NewsArticlePreprocessor] = None
) -> pd.DataFrame:
    """Preprocess an in-memory dataframe of articles.

    Returns a dataframe with the :data:`OUTPUT_COLUMNS`, with rows whose text
    reduces to nothing removed.
    """
    preprocessor = preprocessor or NewsArticlePreprocessor()
    df = _ensure_columns(df)

    combined = (
        df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    )
    df["processed_text"] = preprocessor.preprocess_many(combined.tolist())

    initial_count = len(df)
    df = df[df["processed_text"].str.len() > 0]
    dropped = initial_count - len(df)
    if dropped:
        logger.info("dropped %d article(s) with empty processed text", dropped)

    return df[list(OUTPUT_COLUMNS)].reset_index(drop=True)


def preprocess_dataset(
    input_path: str,
    output_path: str,
    sample_size: Optional[int] = None,
    preprocessor: Optional[NewsArticlePreprocessor] = None,
) -> pd.DataFrame:
    """Load a raw CSV, preprocess it and write the processed CSV.

    Args:
        input_path: Path to the raw CSV file.
        output_path: Where to write the processed CSV.
        sample_size: Optionally sample this many rows (deterministic).
        preprocessor: Custom preprocessor; a default one is built otherwise.

    Returns:
        The processed dataframe.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"raw data not found at {input_path}. Generate a synthetic corpus with "
            f"'python generate_sample_data.py' or drop your own CSV there."
        )

    logger.info("loading dataset from %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("loaded %d articles", len(df))

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logger.info("using a deterministic sample of %d articles", len(df))

    output_df = preprocess_dataframe(df, preprocessor)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("saved %d processed articles to %s", len(output_df), output_path)
    return output_df


def main(argv: Optional[List[str]] = None) -> int:
    from config.config import (
        LOWERCASE,
        MIN_TEXT_LENGTH,
        PROCESSED_DATA_PATH,
        RAW_DATA_PATH,
        REMOVE_PUNCTUATION,
        REMOVE_STOPWORDS,
        SPACY_MODEL,
    )

    parser = argparse.ArgumentParser(description="Preprocess raw news articles.")
    parser.add_argument("--input", default=str(RAW_DATA_PATH), help="raw CSV path")
    parser.add_argument(
        "--output", default=str(PROCESSED_DATA_PATH), help="processed CSV path"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="process a subset only"
    )
    args = parser.parse_args(argv)

    preprocessor = NewsArticlePreprocessor(
        remove_stopwords=REMOVE_STOPWORDS,
        lowercase=LOWERCASE,
        remove_punctuation=REMOVE_PUNCTUATION,
        min_text_length=MIN_TEXT_LENGTH,
        spacy_model=SPACY_MODEL,
    )

    try:
        preprocess_dataset(
            args.input, args.output, sample_size=args.sample_size,
            preprocessor=preprocessor,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    raise SystemExit(main())
