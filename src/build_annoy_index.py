"""Build the nearest-neighbour index from saved embeddings.

The heavy lifting lives in :mod:`src.vector_index`; this module is the CLI /
pipeline step that reads ``embeddings.npy`` plus ``metadata.pkl`` and writes
the index next to them.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Allow both "python -m src.build_annoy_index" and direct script execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.vector_index import (  # noqa: E402
    BaseVectorIndex,
    create_index,
    resolve_backend,
)

logger = logging.getLogger(__name__)


def build_index_from_embeddings(
    embeddings: np.ndarray,
    backend: str = "auto",
    metric: str = "angular",
    num_trees: int = 10,
) -> BaseVectorIndex:
    """Create, populate and build an index for ``embeddings``."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"expected a non-empty 2-D embedding matrix, got shape {embeddings.shape}"
        )

    index = create_index(
        dim=int(embeddings.shape[1]),
        backend=backend,
        metric=metric,
        num_trees=num_trees,
    )
    index.add_items(embeddings)
    index.build()
    return index


def build_annoy_index(
    embeddings_path: str,
    metadata_path: str,
    index_output_path: str,
    num_trees: int = 10,
    metric: str = "angular",
    backend: str = "auto",
) -> Path:
    """Build and save the index described by the pipeline artefacts.

    Args:
        embeddings_path: ``.npy`` matrix written by ``src.sbert_embeddings``.
        metadata_path: ``.pkl`` sidecar written by ``src.sbert_embeddings``.
        index_output_path: Base path for the index; the backend picks the
            extension (``.annoy`` or ``.npz``).
        num_trees: ANNOY tree count (ignored by the exact backend).
        metric: Distance metric.
        backend: ``auto``, ``annoy`` or ``exact``.

    Returns:
        The path the index was written to.
    """
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"embeddings not found at {embeddings_path}. "
            f"Run 'python -m src.sbert_embeddings' first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata not found at {metadata_path}")

    logger.info("loading embeddings from %s", embeddings_path)
    embeddings = np.load(embeddings_path)
    logger.info("loaded embeddings with shape %s", embeddings.shape)

    with open(metadata_path, "rb") as handle:
        metadata = pickle.load(handle)
    num_articles = int(metadata.get("num_articles", len(metadata["article_ids"])))
    logger.info("loaded metadata for %d articles", num_articles)

    expected_dim = int(metadata.get("embedding_dimension") or embeddings.shape[1])
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"dimension mismatch: embeddings have {embeddings.shape[1]} dims, "
            f"metadata expects {expected_dim}"
        )
    if embeddings.shape[0] != num_articles:
        raise ValueError(
            f"row mismatch: {embeddings.shape[0]} embeddings vs "
            f"{num_articles} metadata rows; re-run the embedding step"
        )

    index = build_index_from_embeddings(
        embeddings, backend=backend, metric=metric, num_trees=num_trees
    )
    saved_to = index.save(index_output_path)
    logger.info("%s index built and saved to %s", index.backend, saved_to)
    return saved_to


def main(argv: Optional[List[str]] = None) -> int:
    from config.config import (
        ANNOY_METRIC,
        ANNOY_NUM_TREES,
        EMBEDDINGS_PATH,
        INDEX_BACKEND,
        INDEX_PATH,
        METADATA_PATH,
    )

    parser = argparse.ArgumentParser(description="Build the search index.")
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--output", default=str(INDEX_PATH))
    parser.add_argument("--num-trees", type=int, default=ANNOY_NUM_TREES)
    parser.add_argument("--metric", default=ANNOY_METRIC)
    parser.add_argument(
        "--backend",
        default=INDEX_BACKEND,
        choices=["auto", "annoy", "exact"],
        help="auto prefers annoy when it is installed",
    )
    args = parser.parse_args(argv)

    logger.info("index backend resolved to '%s'", resolve_backend(args.backend))
    try:
        build_annoy_index(
            args.embeddings,
            args.metadata,
            args.output,
            num_trees=args.num_trees,
            metric=args.metric,
            backend=args.backend,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    raise SystemExit(main())
