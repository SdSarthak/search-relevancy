"""Vector index backends used for nearest-neighbour retrieval.

Two interchangeable backends are provided:

``annoy``
    Spotify's ANNOY approximate nearest neighbour index. Fast and memory
    mapped, but it is a C++ extension that needs a compiler toolchain on
    platforms without a prebuilt wheel.

``exact``
    A dependency-free brute-force cosine index backed by a single numpy
    matrix. It returns exact results and is fast enough for corpora up to a
    few hundred thousand articles, so it doubles as the ground truth when
    measuring the recall of the approximate backend.

Both return **cosine similarities in ``[-1, 1]``** so scores are comparable
regardless of which backend served the query.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ("angular",)


def annoy_available() -> bool:
    """Return True when the optional ``annoy`` package can be imported."""
    try:
        import annoy  # noqa: F401
    except ImportError:
        return False
    return True


def angular_distance_to_cosine(distance: float) -> float:
    """Convert an ANNOY angular distance to a cosine similarity.

    ANNOY's angular distance is ``sqrt(2 * (1 - cos))`` for unit vectors, so
    it lives in ``[0, 2]`` and ``cos = 1 - d**2 / 2``. Dividing the distance by
    pi (a common mistake) gives scores that never reach 0 for unrelated
    documents.
    """
    similarity = 1.0 - (float(distance) ** 2) / 2.0
    return float(np.clip(similarity, -1.0, 1.0))


def _as_float32_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {matrix.shape}")
    return matrix


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class BaseVectorIndex:
    """Common interface shared by the index backends."""

    backend = "base"
    file_suffix = ".idx"

    def __init__(self, dim: int, metric: str = "angular", num_trees: int = 10):
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"unsupported metric {metric!r}; supported: {SUPPORTED_METRICS}"
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.metric = metric
        self.num_trees = int(num_trees)
        self.num_items = 0

    def __len__(self) -> int:
        return self.num_items

    def add_items(self, embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def build(self) -> None:
        raise NotImplementedError

    def save(self, path) -> Path:
        raise NotImplementedError

    def query(self, vector: Sequence[float], k: int) -> Tuple[List[int], List[float]]:
        """Return ``(indices, cosine_similarities)`` for the ``k`` best hits."""
        raise NotImplementedError

    def _check_dim(self, matrix: np.ndarray) -> None:
        if matrix.shape[-1] != self.dim:
            raise ValueError(
                f"embedding dimension mismatch: index expects {self.dim}, "
                f"got {matrix.shape[-1]}"
            )


class ExactVectorIndex(BaseVectorIndex):
    """Brute-force cosine index. Exact, pure numpy, no build step required."""

    backend = "exact"
    file_suffix = ".npz"

    def __init__(self, dim: int, metric: str = "angular", num_trees: int = 10):
        super().__init__(dim=dim, metric=metric, num_trees=num_trees)
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    def add_items(self, embeddings: np.ndarray) -> None:
        matrix = _as_float32_matrix(embeddings)
        self._check_dim(matrix)
        matrix = _normalise(matrix)
        self._matrix = (
            matrix if self.num_items == 0 else np.vstack([self._matrix, matrix])
        )
        self.num_items = int(self._matrix.shape[0])
        logger.info("exact index now holds %d vectors", self.num_items)

    def build(self) -> None:  # nothing to precompute
        logger.info("exact index needs no build step (%d vectors)", self.num_items)

    def save(self, path) -> Path:
        target = resolve_index_path(path, self.backend)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, matrix=self._matrix, dim=self.dim, metric=self.metric
        )
        logger.info("saved exact index (%d vectors) to %s", self.num_items, target)
        return target

    @classmethod
    def load(cls, path, dim: int, metric: str = "angular") -> "ExactVectorIndex":
        source = resolve_index_path(path, cls.backend)
        with np.load(source, allow_pickle=False) as payload:
            matrix = payload["matrix"]
            stored_dim = int(payload["dim"])
        if dim and stored_dim != int(dim):
            raise ValueError(
                f"index dimension {stored_dim} does not match expected {dim}"
            )
        index = cls(dim=stored_dim, metric=metric)
        index._matrix = np.asarray(matrix, dtype=np.float32)
        index.num_items = int(index._matrix.shape[0])
        logger.info("loaded exact index (%d vectors) from %s", index.num_items, source)
        return index

    def query(self, vector: Sequence[float], k: int) -> Tuple[List[int], List[float]]:
        if self.num_items == 0:
            return [], []
        query_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self._check_dim(query_vector)
        query_vector = _normalise(query_vector)
        scores = self._matrix @ query_vector
        k = max(1, min(int(k), self.num_items))
        # argpartition gives the top-k cheaply, then sort just those k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [int(i) for i in top], [float(scores[i]) for i in top]


class AnnoyVectorIndex(BaseVectorIndex):
    """ANNOY-backed approximate nearest neighbour index."""

    backend = "annoy"
    file_suffix = ".annoy"

    def __init__(self, dim: int, metric: str = "angular", num_trees: int = 10):
        super().__init__(dim=dim, metric=metric, num_trees=num_trees)
        from annoy import AnnoyIndex  # imported lazily: optional dependency

        self._index = AnnoyIndex(self.dim, metric=self.metric)
        self._built = False

    def add_items(self, embeddings: np.ndarray) -> None:
        if self._built:
            raise RuntimeError("cannot add items to an index that is already built")
        matrix = _as_float32_matrix(embeddings)
        self._check_dim(matrix)
        for offset, embedding in enumerate(matrix):
            self._index.add_item(self.num_items + offset, embedding.tolist())
        self.num_items += int(matrix.shape[0])
        logger.info("annoy index now holds %d vectors", self.num_items)

    def build(self) -> None:
        logger.info("building annoy index with %d trees", self.num_trees)
        self._index.build(self.num_trees)
        self._built = True

    def save(self, path) -> Path:
        if not self._built:
            self.build()
        target = resolve_index_path(path, self.backend)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._index.save(str(target))
        logger.info("saved annoy index (%d vectors) to %s", self.num_items, target)
        return target

    @classmethod
    def load(cls, path, dim: int, metric: str = "angular") -> "AnnoyVectorIndex":
        source = resolve_index_path(path, cls.backend)
        index = cls(dim=dim, metric=metric)
        index._index.load(str(source))
        index.num_items = index._index.get_n_items()
        index._built = True
        logger.info("loaded annoy index (%d vectors) from %s", index.num_items, source)
        return index

    def query(self, vector: Sequence[float], k: int) -> Tuple[List[int], List[float]]:
        if self.num_items == 0:
            return [], []
        query_vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        self._check_dim(query_vector)
        k = max(1, min(int(k), self.num_items))
        indices, distances = self._index.get_nns_by_vector(
            query_vector.tolist(), k, include_distances=True
        )
        return (
            [int(i) for i in indices],
            [angular_distance_to_cosine(d) for d in distances],
        )


_BACKENDS = {
    ExactVectorIndex.backend: ExactVectorIndex,
    AnnoyVectorIndex.backend: AnnoyVectorIndex,
}


def resolve_backend(backend: str = "auto") -> str:
    """Resolve ``"auto"`` to a concrete backend name."""
    backend = (backend or "auto").strip().lower()
    if backend == "auto":
        return "annoy" if annoy_available() else "exact"
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown index backend {backend!r}; choose from "
            f"{sorted(_BACKENDS) + ['auto']}"
        )
    if backend == "annoy" and not annoy_available():
        raise ImportError(
            "INDEX_BACKEND=annoy but the 'annoy' package is not installed. "
            "Install it with 'pip install -r requirements-annoy.txt' (needs a "
            "C++ toolchain) or use INDEX_BACKEND=exact."
        )
    return backend


def resolve_index_path(path, backend: str) -> Path:
    """Give each backend its own on-disk extension, sharing one base path."""
    suffix = _BACKENDS[resolve_backend(backend)].file_suffix
    return Path(path).with_suffix(suffix)


def create_index(
    dim: int,
    backend: str = "auto",
    metric: str = "angular",
    num_trees: int = 10,
) -> BaseVectorIndex:
    """Create an empty index using the requested (or best available) backend."""
    resolved = resolve_backend(backend)
    logger.info("using '%s' index backend (dim=%d, metric=%s)", resolved, dim, metric)
    return _BACKENDS[resolved](dim=dim, metric=metric, num_trees=num_trees)


def load_index(
    path, dim: int, backend: str = "auto", metric: str = "angular"
) -> BaseVectorIndex:
    """Load a previously saved index.

    With ``backend="auto"`` an ANNOY index is preferred when both the package
    and an ``.annoy`` file are present, otherwise the exact ``.npz`` index is
    used.
    """
    requested = (backend or "auto").strip().lower()
    if requested == "auto":
        candidates = ["annoy", "exact"] if annoy_available() else ["exact"]
        for candidate in candidates:
            if resolve_index_path(path, candidate).exists():
                requested = candidate
                break
        else:
            searched = ", ".join(
                str(resolve_index_path(path, c)) for c in candidates
            )
            raise FileNotFoundError(f"no index found (looked for: {searched})")
    resolved = resolve_backend(requested)
    return _BACKENDS[resolved].load(path, dim=dim, metric=metric)
