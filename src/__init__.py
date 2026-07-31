"""Semantic search over news articles: preprocessing, embedding, indexing, API."""

import sys
from pathlib import Path

# Make the project root importable so "config.config" resolves however the
# package is entered (module, script or WSGI server).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

__version__ = "1.1.0"
