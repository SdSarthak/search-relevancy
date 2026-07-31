"""Measure search relevancy.

Two questions get answered here:

1. **Is the ranking any good?** Precision@k, Recall@k, MRR and nDCG@k against
   a labelled query set.
2. **What does the approximation cost?** The overlap between the ANNOY top-k
   and the exact brute-force top-k, i.e. how much recall is traded for speed.

Ground truth can either be supplied (``--queries queries.json``) or derived
from the corpus itself: each article's title becomes a query whose only
relevant document is that article (known-item retrieval). The derived set is
a proxy, but it is reproducible and needs no manual labelling.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# Allow both "python -m src.evaluate" and direct script execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.search_engine import SearchEngine  # noqa: E402
from src.vector_index import BaseVectorIndex, ExactVectorIndex  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES = (1, 5, 10)


# --------------------------------------------------------------------------
# Ranking metrics (pure functions, easy to unit test)
# --------------------------------------------------------------------------
def precision_at_k(ranked: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / float(k)


def recall_at_k(ranked: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Fraction of the relevant documents that appear in the top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / float(len(relevant))


def reciprocal_rank(ranked: Sequence[Any], relevant: Iterable[Any]) -> float:
    """1 / rank of the first relevant hit (0 when there is none)."""
    relevant = set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    gains = list(gains)[:k]
    return float(
        sum(gain / np.log2(position + 1) for position, gain in enumerate(gains, start=1))
    )


def ndcg_at_k(ranked: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant)
    if not relevant:
        return 0.0
    gains = [1.0 if item in relevant else 0.0 for item in ranked[:k]]
    ideal = [1.0] * min(len(relevant), k)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def overlap_at_k(approximate: Sequence[Any], exact: Sequence[Any], k: int) -> float:
    """Share of the exact top-k that the approximate search also returned."""
    if k <= 0:
        raise ValueError("k must be positive")
    exact_top = list(exact)[:k]
    if not exact_top:
        return 0.0
    return len(set(approximate[:k]) & set(exact_top)) / float(len(exact_top))


# --------------------------------------------------------------------------
# Query sets
# --------------------------------------------------------------------------
def build_known_item_queries(
    metadata: Dict[str, Any],
    sample_size: Optional[int] = 200,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Derive a query set from article titles (known-item retrieval)."""
    titles = metadata.get("titles") or []
    article_ids = metadata.get("article_ids") or []
    categories = metadata.get("categories") or []

    candidates = [
        {
            "query": str(title).strip(),
            "relevant_ids": [article_ids[position]],
            "category": categories[position] if position < len(categories) else None,
        }
        for position, title in enumerate(titles)
        if position < len(article_ids) and str(title).strip()
    ]
    if sample_size and sample_size < len(candidates):
        candidates = random.Random(seed).sample(candidates, sample_size)
    return candidates


def load_query_set(path) -> List[Dict[str, Any]]:
    """Load a labelled query set from JSON.

    Expected shape::

        [{"query": "climate change", "relevant_ids": ["article_00001"]}, ...]
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("queries", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} does not contain a non-empty list of queries")

    queries = []
    for entry in payload:
        if not isinstance(entry, dict) or "query" not in entry:
            raise ValueError(f"malformed query entry in {path}: {entry!r}")
        relevant = entry.get("relevant_ids", entry.get("relevant", []))
        if isinstance(relevant, (str, int)):
            relevant = [relevant]
        queries.append(
            {
                "query": str(entry["query"]),
                "relevant_ids": list(relevant),
                "category": entry.get("category"),
            }
        )
    return queries


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def evaluate_ranking(
    engine: SearchEngine,
    queries: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    """Run the query set through ``engine`` and aggregate ranking metrics."""
    if not queries:
        raise ValueError("query set is empty")
    k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not k_values:
        raise ValueError("k_values must contain at least one positive integer")
    top_k = max(k_values)

    per_k = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    reciprocal_ranks: List[float] = []
    category_hits: List[float] = []
    latencies_ms: List[float] = []

    for entry in queries:
        started = time.perf_counter()
        response = engine.search(entry["query"], top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        ranked_ids = [hit["article_id"] for hit in response["results"]]
        relevant = entry.get("relevant_ids") or []
        for k in k_values:
            per_k[k]["precision"].append(precision_at_k(ranked_ids, relevant, k))
            per_k[k]["recall"].append(recall_at_k(ranked_ids, relevant, k))
            per_k[k]["ndcg"].append(ndcg_at_k(ranked_ids, relevant, k))
        reciprocal_ranks.append(reciprocal_rank(ranked_ids, relevant))

        expected_category = entry.get("category")
        if expected_category and response["results"]:
            category_hits.append(
                1.0
                if response["results"][0].get("category") == expected_category
                else 0.0
            )

    report: Dict[str, Any] = {
        "num_queries": len(queries),
        "k_values": k_values,
        "metrics": {
            f"@{k}": {
                "precision": round(float(np.mean(per_k[k]["precision"])), 4),
                "recall": round(float(np.mean(per_k[k]["recall"])), 4),
                "ndcg": round(float(np.mean(per_k[k]["ndcg"])), 4),
            }
            for k in k_values
        },
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 3),
            "p50": round(_percentile(latencies_ms, 50), 3),
            "p95": round(_percentile(latencies_ms, 95), 3),
        },
    }
    if category_hits:
        report["top1_category_match"] = round(float(np.mean(category_hits)), 4)
    return report


def evaluate_approximation(
    index: BaseVectorIndex,
    embeddings: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 10,
) -> Dict[str, Any]:
    """Compare the live index against exact brute-force search.

    Returns the mean top-k overlap plus both engines' latencies. When the live
    index is already exact the overlap is 1.0 by construction, which makes the
    check a useful regression guard either way.
    """
    exact = ExactVectorIndex(dim=int(embeddings.shape[1]))
    exact.add_items(embeddings)

    overlaps: List[float] = []
    approx_ms: List[float] = []
    exact_ms: List[float] = []

    for vector in np.asarray(query_vectors, dtype=np.float32):
        started = time.perf_counter()
        approximate_ids, _ = index.query(vector, k)
        approx_ms.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        exact_ids, _ = exact.query(vector, k)
        exact_ms.append((time.perf_counter() - started) * 1000.0)

        overlaps.append(overlap_at_k(approximate_ids, exact_ids, k))

    return {
        "backend": index.backend,
        "k": int(k),
        "num_queries": int(len(overlaps)),
        f"recall_vs_exact@{k}": round(float(np.mean(overlaps)), 4) if overlaps else 0.0,
        "latency_ms": {
            "approximate_mean": round(float(np.mean(approx_ms)), 3) if approx_ms else 0.0,
            "exact_mean": round(float(np.mean(exact_ms)), 3) if exact_ms else 0.0,
        },
    }


def format_report(report: Dict[str, Any]) -> str:
    """Render an evaluation report as a readable text block."""
    lines = ["Search relevancy report", "=" * 40]
    ranking = report.get("ranking")
    if ranking:
        lines.append(f"queries evaluated : {ranking['num_queries']}")
        lines.append(f"MRR               : {ranking['mrr']:.4f}")
        for label, values in ranking["metrics"].items():
            lines.append(
                f"  P{label:<5} {values['precision']:.4f}   "
                f"R{label:<5} {values['recall']:.4f}   "
                f"nDCG{label:<5} {values['ndcg']:.4f}"
            )
        if "top1_category_match" in ranking:
            lines.append(
                f"top-1 category match: {ranking['top1_category_match']:.4f}"
            )
        latency = ranking["latency_ms"]
        lines.append(
            f"latency ms        : mean {latency['mean']:.2f} "
            f"p50 {latency['p50']:.2f} p95 {latency['p95']:.2f}"
        )

    approximation = report.get("approximation")
    if approximation:
        k = approximation["k"]
        lines.append("-" * 40)
        lines.append(f"index backend     : {approximation['backend']}")
        lines.append(
            f"recall vs exact@{k}: {approximation[f'recall_vs_exact@{k}']:.4f}"
        )
        lines.append(
            "latency ms        : approx "
            f"{approximation['latency_ms']['approximate_mean']:.2f} "
            f"exact {approximation['latency_ms']['exact_mean']:.2f}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    from config.config import (
        ANNOY_METRIC,
        EMBEDDINGS_PATH,
        INDEX_BACKEND,
        INDEX_PATH,
        MAX_NUM_RESULTS,
        METADATA_PATH,
        SBERT_MODEL,
    )
    from src.search_engine import load_search_engine

    parser = argparse.ArgumentParser(description="Evaluate search relevancy.")
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--backend", default=INDEX_BACKEND)
    parser.add_argument(
        "--queries", default=None, help="JSON file with a labelled query set"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="queries to derive from titles when --queries is not given",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES), help="cut-offs"
    )
    parser.add_argument("--report", default=None, help="write the report as JSON here")
    args = parser.parse_args(argv)

    try:
        engine = load_search_engine(
            index_path=args.index,
            metadata_path=args.metadata,
            model_name=SBERT_MODEL,
            backend=args.backend,
            metric=ANNOY_METRIC,
            max_num_results=max(MAX_NUM_RESULTS, max(args.k)),
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        return 1

    if args.queries:
        queries = load_query_set(args.queries)
    else:
        queries = build_known_item_queries(engine.metadata, sample_size=args.sample_size)
        logger.info("derived %d known-item queries from article titles", len(queries))

    report: Dict[str, Any] = {
        "index_backend": engine.index.backend,
        "embedding_model": engine.metadata.get("embedding_model"),
        "num_articles": engine.num_articles,
        "ranking": evaluate_ranking(engine, queries, args.k),
    }

    embeddings_path = Path(args.embeddings)
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        query_vectors = np.vstack(
            [engine.encode_query(entry["query"]) for entry in queries[:100]]
        )
        report["approximation"] = evaluate_approximation(
            engine.index, embeddings, query_vectors, k=max(args.k)
        )
    else:
        logger.warning(
            "embeddings not found at %s; skipping the approximation check",
            embeddings_path,
        )

    print(format_report(report))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("wrote JSON report to %s", report_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    raise SystemExit(main())
