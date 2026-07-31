"""Generate SBERT embeddings and the metadata sidecar for processed articles."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow both "python -m src.sbert_embeddings" and direct script execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

#: Article body characters retained in the metadata sidecar. Keeping whole
#: bodies makes the pickle as large as the corpus for no retrieval benefit.
METADATA_TEXT_CHARS = 1000


class SBERTEmbedder:
    """Generate SBERT embeddings for a collection of texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info("loading SBERT model: %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
        logger.info("model loaded, embedding dimension: %d", self.embedding_dim)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode ``texts`` into a ``(len(texts), dim)`` float32 array."""
        logger.info("encoding %d texts", len(texts))
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        logger.info("encoding complete, shape: %s", embeddings.shape)
        return embeddings


def _column(df: pd.DataFrame, name: str) -> List[Any]:
    if name not in df.columns:
        return []
    return df[name].where(df[name].notna(), None).tolist()


def build_metadata(
    df: pd.DataFrame,
    model_name: str,
    embedding_dim: int,
    text_chars: int = METADATA_TEXT_CHARS,
) -> Dict[str, Any]:
    """Build the metadata sidecar mapping index positions back to articles."""
    texts = _column(df, "text")
    if text_chars > 0:
        texts = [None if text is None else str(text)[:text_chars] for text in texts]

    return {
        "article_ids": _column(df, "article_id"),
        "titles": _column(df, "title"),
        "categories": _column(df, "category"),
        "subcategories": _column(df, "subcategory"),
        "sources": _column(df, "source"),
        "published_dates": _column(df, "published_date"),
        "texts": texts,
        "embedding_model": model_name,
        "embedding_dimension": int(embedding_dim),
        "num_articles": int(len(df)),
    }


def texts_to_embed(df: pd.DataFrame) -> List[str]:
    """Pick what to embed: the processed text when it is populated."""
    if "processed_text" in df.columns:
        column = df["processed_text"].fillna("").astype(str)
        if column.str.len().gt(0).any():
            return column.tolist()
        logger.warning("processed_text column is empty; falling back to title+text")

    if "title" not in df.columns and "text" not in df.columns:
        raise ValueError("input data has neither 'processed_text' nor 'title'/'text'")

    empty = pd.Series([""] * len(df), index=df.index)
    title = df["title"].fillna("").astype(str) if "title" in df.columns else empty
    body = df["text"].fillna("").astype(str) if "text" in df.columns else empty
    return (title + " " + body).str.strip().tolist()


def generate_embeddings(
    input_path: str,
    embeddings_output_path: str,
    metadata_output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    embedder: Optional[SBERTEmbedder] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Embed a processed CSV and persist the vectors plus metadata.

    Args:
        input_path: Processed CSV produced by ``src.data_preprocessing``.
        embeddings_output_path: Where to write the ``.npy`` matrix.
        metadata_output_path: Where to write the ``.pkl`` metadata sidecar.
        model_name: SBERT model to load when ``embedder`` is not supplied.
        batch_size: Encoder batch size.
        embedder: Pre-built embedder (used by tests).

    Returns:
        ``(embeddings, metadata)``.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"processed data not found at {input_path}. "
            f"Run 'python -m src.data_preprocessing' first."
        )

    logger.info("loading processed data from %s", input_path)
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"{input_path} contains no rows")
    logger.info("loaded %d articles", len(df))

    embedder = embedder or SBERTEmbedder(model_name=model_name)
    embeddings = embedder.encode(texts_to_embed(df), batch_size=batch_size)

    metadata = build_metadata(
        df, model_name=embedder.model_name, embedding_dim=embedder.embedding_dim
    )

    embeddings_output_path = Path(embeddings_output_path)
    embeddings_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_output_path, embeddings)
    logger.info("saved embeddings to %s", embeddings_output_path)

    metadata_output_path = Path(metadata_output_path)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "wb") as handle:
        pickle.dump(metadata, handle)
    logger.info("saved metadata to %s", metadata_output_path)

    return embeddings, metadata


def main(argv: Optional[List[str]] = None) -> int:
    from config.config import (
        EMBEDDING_BATCH_SIZE,
        EMBEDDINGS_PATH,
        METADATA_PATH,
        PROCESSED_DATA_PATH,
        SBERT_MODEL,
    )

    parser = argparse.ArgumentParser(description="Generate SBERT embeddings.")
    parser.add_argument("--input", default=str(PROCESSED_DATA_PATH))
    parser.add_argument("--embeddings-output", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--metadata-output", default=str(METADATA_PATH))
    parser.add_argument("--model", default=SBERT_MODEL)
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    args = parser.parse_args(argv)

    try:
        generate_embeddings(
            args.input,
            args.embeddings_output,
            args.metadata_output,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    raise SystemExit(main())
