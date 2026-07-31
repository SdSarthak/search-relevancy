"""Query-time search engine.

``SearchEngine`` ties together an encoder (SBERT by default), a vector index
and the article metadata. It deliberately knows nothing about Flask so it can
be exercised from tests, notebooks or a CLI with a stub encoder and without
downloading a model.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Allow both "python -m src.search_engine" and "python src/search_engine.py".
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.vector_index import BaseVectorIndex, load_index  # noqa: E402

logger = logging.getLogger(__name__)

# Metadata keys holding one entry per indexed article.
_PER_ARTICLE_FIELDS = (
    "article_ids",
    "titles",
    "categories",
    "subcategories",
    "sources",
    "published_dates",
    "texts",
)


class SBERTEncoder:
    """Thin wrapper around ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info("loading SBERT model %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


def load_metadata(path) -> Dict[str, Any]:
    """Load the pickled metadata written by the embedding step."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"metadata not found at {path}. Run 'python -m src.sbert_embeddings' first."
        )
    with open(path, "rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, dict) or "article_ids" not in metadata:
        raise ValueError(f"{path} does not look like search-relevancy metadata")
    metadata.setdefault("num_articles", len(metadata["article_ids"]))
    return metadata


def snippet(text: Optional[str], max_chars: int) -> Optional[str]:
    """Trim an article body to ``max_chars`` (0 or less keeps everything)."""
    if text is None:
        return None
    text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


class SearchEngine:
    """Encode a query, retrieve neighbours and decorate them with metadata."""

    def __init__(
        self,
        encoder: Any,
        index: BaseVectorIndex,
        metadata: Dict[str, Any],
        default_num_results: int = 10,
        max_num_results: int = 50,
        snippet_chars: int = 400,
    ):
        self.encoder = encoder
        self.index = index
        self.metadata = metadata
        self.default_num_results = int(default_num_results)
        self.max_num_results = int(max_num_results)
        self.snippet_chars = int(snippet_chars)

        num_articles = int(metadata.get("num_articles", len(index)))
        if len(index) and num_articles != len(index):
            raise ValueError(
                f"index holds {len(index)} vectors but metadata describes "
                f"{num_articles} articles; rebuild the index"
            )
        self.num_articles = num_articles

    # -- helpers -----------------------------------------------------------
    def _field(self, name: str, position: int):
        values = self.metadata.get(name)
        if not values or position >= len(values):
            return None
        value = values[position]
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    def clamp_num_results(self, num_results: Optional[int]) -> int:
        if num_results is None:
            num_results = self.default_num_results
        try:
            num_results = int(num_results)
        except (TypeError, ValueError):
            raise ValueError("num_results must be an integer")
        return max(1, min(num_results, self.max_num_results))

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.encoder.encode([query])
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

    # -- public API --------------------------------------------------------
    def search(
        self, query: str, num_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """Return the most semantically similar articles for ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        num_results = self.clamp_num_results(num_results)

        query_embedding = self.encode_query(query)
        indices, similarities = self.index.query(query_embedding, num_results)

        results: List[Dict[str, Any]] = []
        for position, score in zip(indices, similarities):
            results.append(
                {
                    "article_id": self._field("article_ids", position),
                    "title": self._field("titles", position),
                    "category": self._field("categories", position),
                    "subcategory": self._field("subcategories", position),
                    "source": self._field("sources", position),
                    "published_date": self._field("published_dates", position),
                    "text": snippet(
                        self._field("texts", position), self.snippet_chars
                    ),
                    "relevance_score": round(float(score), 6),
                }
            )

        return {"query": query, "num_results": len(results), "results": results}

    def batch_search(
        self, queries: Sequence[str], num_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search several queries, skipping blank ones."""
        return [
            self.search(query, num_results)
            for query in queries
            if isinstance(query, str) and query.strip()
        ]

    def info(self) -> Dict[str, Any]:
        return {
            "num_articles": self.num_articles,
            "embedding_model": self.metadata.get("embedding_model"),
            "embedding_dimension": self.metadata.get(
                "embedding_dimension", self.index.dim
            ),
            "index_backend": self.index.backend,
            "default_results": self.default_num_results,
            "max_results": self.max_num_results,
        }


def load_search_engine(
    index_path,
    metadata_path,
    model_name: str = "all-MiniLM-L6-v2",
    backend: str = "auto",
    metric: str = "angular",
    encoder: Any = None,
    default_num_results: int = 10,
    max_num_results: int = 50,
    snippet_chars: int = 400,
) -> SearchEngine:
    """Build a ``SearchEngine`` from artefacts on disk.

    Pass ``encoder`` to skip loading SBERT (used by the tests).
    """
    metadata = load_metadata(metadata_path)
    dim = int(metadata.get("embedding_dimension") or 0)

    if encoder is None:
        encoder = SBERTEncoder(metadata.get("embedding_model") or model_name)
        if dim and encoder.dim != dim:
            raise ValueError(
                f"model '{encoder.model_name}' produces {encoder.dim}-dim vectors "
                f"but the index was built with {dim} dims; rebuild the index or "
                f"set SBERT_MODEL to the original model"
            )
        dim = dim or encoder.dim

    index = load_index(index_path, dim=dim, backend=backend, metric=metric)
    return SearchEngine(
        encoder=encoder,
        index=index,
        metadata=metadata,
        default_num_results=default_num_results,
        max_num_results=max_num_results,
        snippet_chars=snippet_chars,
    )
