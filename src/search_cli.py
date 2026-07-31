"""Query the index from a terminal, without starting the API.

    python -m src.search_cli "climate change" -n 5
    python -m src.search_cli            # interactive prompt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Allow both "python -m src.search_cli" and direct script execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.search_engine import SearchEngine, load_search_engine  # noqa: E402

logger = logging.getLogger(__name__)


def format_results(response: dict, snippet_chars: int = 160) -> str:
    """Render a search response as a numbered list."""
    lines = [f"\n{response['num_results']} result(s) for {response['query']!r}"]
    for rank, hit in enumerate(response["results"], start=1):
        lines.append(
            f"{rank:>3}. [{hit['relevance_score']:+.4f}] {hit['title']}"
        )
        meta = " / ".join(
            str(value)
            for value in (hit.get("category"), hit.get("source"), hit.get("published_date"))
            if value
        )
        if meta:
            lines.append(f"     {meta}")
        text = (hit.get("text") or "").replace("\n", " ")
        if text:
            lines.append(f"     {text[:snippet_chars]}")
    return "\n".join(lines)


def run_interactive(engine: SearchEngine, num_results: int) -> int:
    print("Type a query and press enter. Blank line or Ctrl-C to quit.")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            return 0
        try:
            print(format_results(engine.search(query, num_results)))
        except ValueError as exc:
            print(f"error: {exc}")


def main(argv: Optional[List[str]] = None) -> int:
    from config.config import (
        ANNOY_METRIC,
        DEFAULT_NUM_RESULTS,
        INDEX_BACKEND,
        INDEX_PATH,
        MAX_NUM_RESULTS,
        METADATA_PATH,
        SBERT_MODEL,
        SNIPPET_CHARS,
    )

    parser = argparse.ArgumentParser(description="Search the article index.")
    parser.add_argument("query", nargs="*", help="query text; omit for a prompt")
    parser.add_argument("-n", "--num-results", type=int, default=DEFAULT_NUM_RESULTS)
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--backend", default=INDEX_BACKEND)
    args = parser.parse_args(argv)

    try:
        engine = load_search_engine(
            index_path=args.index,
            metadata_path=args.metadata,
            model_name=SBERT_MODEL,
            backend=args.backend,
            metric=ANNOY_METRIC,
            default_num_results=DEFAULT_NUM_RESULTS,
            max_num_results=MAX_NUM_RESULTS,
            snippet_chars=SNIPPET_CHARS,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        return 1

    if not args.query:
        return run_interactive(engine, args.num_results)

    try:
        print(format_results(engine.search(" ".join(args.query), args.num_results)))
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
