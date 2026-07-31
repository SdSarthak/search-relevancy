"""Flask REST API exposing the semantic article search.

Create the WSGI application with :func:`create_app` so it works both with the
built-in dev server (``python -m src.app``) and a production server, e.g.::

    gunicorn -w 2 -b 0.0.0.0:5000 "src.app:create_app()"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

# Allow both "python -m src.app" and direct script execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.config import (  # noqa: E402
    ANNOY_METRIC,
    DEFAULT_NUM_RESULTS,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    INDEX_BACKEND,
    INDEX_PATH,
    MAX_NUM_RESULTS,
    METADATA_PATH,
    SBERT_MODEL,
    SNIPPET_CHARS,
)
from src.search_engine import SearchEngine, load_search_engine  # noqa: E402

logger = logging.getLogger(__name__)

MIN_QUERY_LENGTH = 2
MAX_QUERY_LENGTH = 512
MAX_BATCH_QUERIES = 25


def build_default_engine() -> SearchEngine:
    """Load the search engine from the configured artefact paths."""
    return load_search_engine(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        model_name=SBERT_MODEL,
        backend=INDEX_BACKEND,
        metric=ANNOY_METRIC,
        default_num_results=DEFAULT_NUM_RESULTS,
        max_num_results=MAX_NUM_RESULTS,
        snippet_chars=SNIPPET_CHARS,
    )


def _validate_query(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ValueError("query must be a string")
    query = raw.strip()
    if len(query) < MIN_QUERY_LENGTH:
        raise ValueError(
            f"query must be at least {MIN_QUERY_LENGTH} characters long"
        )
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters long")
    return query


def create_app(engine: Optional[SearchEngine] = None, eager: bool = True) -> Flask:
    """Build the Flask application.

    Args:
        engine: Pre-built search engine. When omitted one is loaded from disk.
        eager: Load the engine at start-up. Set ``False`` to defer loading to
            the first request (handy when the container starts before the
            model volume is mounted).
    """
    app = Flask(__name__)
    CORS(app)
    app.config["ENGINE"] = engine
    app.config["ENGINE_ERROR"] = None

    def get_engine() -> Optional[SearchEngine]:
        if app.config["ENGINE"] is None and app.config["ENGINE_ERROR"] is None:
            try:
                app.config["ENGINE"] = build_default_engine()
            except Exception as exc:  # surfaced through /health
                app.config["ENGINE_ERROR"] = str(exc)
                logger.error("failed to load search engine: %s", exc)
        return app.config["ENGINE"]

    def require_engine():
        engine = get_engine()
        if engine is None:
            detail = app.config["ENGINE_ERROR"] or "search engine not initialised"
            return None, (
                jsonify({"error": "Service unavailable", "details": detail}),
                503,
            )
        return engine, None

    if eager and engine is None:
        get_engine()

    @app.get("/health")
    def health_check():
        engine = get_engine()
        if engine is None:
            return (
                jsonify(
                    {
                        "status": "unhealthy",
                        "service": "Search Relevancy API",
                        "details": app.config["ENGINE_ERROR"]
                        or "search engine not initialised",
                    }
                ),
                503,
            )
        return (
            jsonify(
                {
                    "status": "healthy",
                    "service": "Search Relevancy API",
                    "num_articles": engine.num_articles,
                }
            ),
            200,
        )

    @app.get("/info")
    def info():
        engine, failure = require_engine()
        if failure:
            return failure
        payload = {"service": "Search Relevancy API"}
        payload.update(engine.info())
        return jsonify(payload), 200

    @app.post("/search")
    def search():
        """Search endpoint.

        Body: ``{"query": "climate change", "num_results": 10}``
        """
        engine, failure = require_engine()
        if failure:
            return failure

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "query" not in data:
            return jsonify({"error": "Missing required field: query"}), 400

        try:
            query = _validate_query(data["query"])
            num_results = engine.clamp_num_results(
                data.get("num_results", DEFAULT_NUM_RESULTS)
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            results = engine.search(query, num_results)
        except Exception as exc:
            logger.exception("search failed for query %r", query)
            return jsonify({"error": "Internal server error", "details": str(exc)}), 500
        return jsonify(results), 200

    @app.post("/search/batch")
    def batch_search():
        """Batch endpoint.

        Body: ``{"queries": ["a", "b"], "num_results": 5}``
        """
        engine, failure = require_engine()
        if failure:
            return failure

        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "queries" not in data:
            return jsonify({"error": "Missing required field: queries"}), 400

        queries = data["queries"]
        if not isinstance(queries, list) or not queries:
            return jsonify({"error": "queries must be a non-empty list"}), 400
        if len(queries) > MAX_BATCH_QUERIES:
            return (
                jsonify({"error": f"at most {MAX_BATCH_QUERIES} queries per batch"}),
                400,
            )

        try:
            num_results = engine.clamp_num_results(
                data.get("num_results", DEFAULT_NUM_RESULTS)
            )
            validated = [_validate_query(query) for query in queries]
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            batch_results = engine.batch_search(validated, num_results)
        except Exception as exc:
            logger.exception("batch search failed")
            return jsonify({"error": "Internal server error", "details": str(exc)}), 500
        return jsonify({"batch_results": batch_results}), 200

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("internal server error: %s", error)
        return jsonify({"error": "Internal server error"}), 500

    return app


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app = create_app()
    logger.info("starting Flask server on %s:%s", FLASK_HOST, FLASK_PORT)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
