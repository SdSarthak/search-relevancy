"""Shared fixtures.

Everything here is synthetic and deterministic: no network, no model
downloads, no database. The stub encoder replaces SBERT with a hashing
bag-of-words so tests still exercise real ranking behaviour.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search_engine import SearchEngine  # noqa: E402
from src.vector_index import ExactVectorIndex  # noqa: E402

STUB_DIM = 32


class HashingEncoder:
    """Deterministic stand-in for SBERT.

    Hashes tokens into a fixed number of buckets, so documents sharing words
    end up close together under cosine similarity. ``zlib.crc32`` is used
    instead of ``hash()`` because the latter is salted per process.
    """

    def __init__(self, dim: int = STUB_DIM):
        self.dim = dim
        self.model_name = "stub-hashing-encoder"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).lower().split():
                bucket = zlib.crc32(token.encode("utf-8")) % self.dim
                vectors[row, bucket] += 1.0
        return vectors


#: A tiny corpus with three clearly separated topics.
CORPUS: List[dict] = [
    {
        "article_id": "article_00001",
        "category": "Science",
        "subcategory": "Environment",
        "title": "climate warming oceans",
        "published_date": "2024-01-01",
        "source": "Nature",
        "text": "climate warming oceans rise steadily across every measured basin",
    },
    {
        "article_id": "article_00002",
        "category": "Science",
        "subcategory": "Environment",
        "title": "climate policy summit",
        "published_date": "2024-01-02",
        "source": "Reuters",
        "text": "climate policy summit delegates debate emissions targets",
    },
    {
        "article_id": "article_00003",
        "category": "Technology",
        "subcategory": "AI",
        "title": "neural networks training",
        "published_date": "2024-02-01",
        "source": "TechCrunch",
        "text": "neural networks training throughput doubles on newer accelerators",
    },
    {
        "article_id": "article_00004",
        "category": "Technology",
        "subcategory": "Software",
        "title": "compiler optimisation release",
        "published_date": "2024-02-02",
        "source": "TechCrunch",
        "text": "compiler optimisation release shortens build times for large projects",
    },
    {
        "article_id": "article_00005",
        "category": "Business",
        "subcategory": "Markets",
        "title": "markets rally quarterly earnings",
        "published_date": "2024-03-01",
        "source": "BBC",
        "text": "markets rally quarterly earnings beat analyst expectations",
    },
]


@pytest.fixture
def encoder() -> HashingEncoder:
    return HashingEncoder()


@pytest.fixture
def corpus() -> List[dict]:
    return [dict(article) for article in CORPUS]


@pytest.fixture
def metadata(corpus) -> dict:
    return {
        "article_ids": [a["article_id"] for a in corpus],
        "titles": [a["title"] for a in corpus],
        "categories": [a["category"] for a in corpus],
        "subcategories": [a["subcategory"] for a in corpus],
        "sources": [a["source"] for a in corpus],
        "published_dates": [a["published_date"] for a in corpus],
        "texts": [a["text"] for a in corpus],
        "embedding_model": "stub-hashing-encoder",
        "embedding_dimension": STUB_DIM,
        "num_articles": len(corpus),
    }


@pytest.fixture
def embeddings(encoder, corpus) -> np.ndarray:
    return encoder.encode([f"{a['title']} {a['text']}" for a in corpus])


@pytest.fixture
def exact_index(embeddings) -> ExactVectorIndex:
    index = ExactVectorIndex(dim=STUB_DIM)
    index.add_items(embeddings)
    index.build()
    return index


@pytest.fixture
def engine(encoder, exact_index, metadata) -> SearchEngine:
    return SearchEngine(
        encoder=encoder,
        index=exact_index,
        metadata=metadata,
        default_num_results=3,
        max_num_results=5,
        snippet_chars=20,
    )
