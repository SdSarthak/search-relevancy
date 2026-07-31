"""Central configuration.

Every value can be overridden with an environment variable. A local ``.env``
file (see ``.env.example``) is loaded automatically when python-dotenv is
installed, so nothing here needs to be edited to run the project.
"""

import os
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - trivial fallback
    def load_dotenv(*_args, **_kwargs):
        return False

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

load_dotenv(BASE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable.

    ``os.getenv("X", False)`` returns the *string* ``"0"`` when the variable is
    set to ``0``, and a non-empty string is truthy. This helper avoids that.
    """
    return _env_str(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return default if not value else Path(value)


# Data / artefact locations
DATA_DIR = _env_path("DATA_DIR", BASE_DIR / "data")
MODEL_DIR = _env_path("MODEL_DIR", BASE_DIR / "models")

RAW_DATA_PATH = _env_path("RAW_DATA_PATH", DATA_DIR / "raw" / "news_articles.csv")
PROCESSED_DATA_PATH = _env_path(
    "PROCESSED_DATA_PATH", DATA_DIR / "processed" / "processed_articles.csv"
)
EMBEDDINGS_PATH = _env_path("EMBEDDINGS_PATH", MODEL_DIR / "embeddings.npy")
METADATA_PATH = _env_path("METADATA_PATH", MODEL_DIR / "metadata.pkl")

# SBERT configuration
SBERT_MODEL = _env_str("SBERT_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)

# Index configuration.
# "auto" uses ANNOY when it is importable and falls back to the exact numpy
# backend otherwise. Force one with INDEX_BACKEND=annoy|exact.
INDEX_BACKEND = _env_str("INDEX_BACKEND", "auto").strip().lower()
INDEX_PATH = _env_path("INDEX_PATH", MODEL_DIR / "articles_index.annoy")
ANNOY_INDEX_PATH = INDEX_PATH  # backwards-compatible alias
ANNOY_NUM_TREES = _env_int("ANNOY_NUM_TREES", 10)
ANNOY_METRIC = _env_str("ANNOY_METRIC", "angular")  # angular == cosine

# Flask configuration
FLASK_HOST = _env_str("FLASK_HOST", "0.0.0.0")
FLASK_PORT = _env_int("FLASK_PORT", 5000)
FLASK_DEBUG = _env_bool("FLASK_DEBUG", False)

# Search configuration
DEFAULT_NUM_RESULTS = _env_int("DEFAULT_NUM_RESULTS", 10)
MAX_NUM_RESULTS = _env_int("MAX_NUM_RESULTS", 50)
# Characters of article body returned per hit (0 returns the full text).
SNIPPET_CHARS = _env_int("SNIPPET_CHARS", 400)

# Preprocessing configuration
MIN_TEXT_LENGTH = _env_int("MIN_TEXT_LENGTH", 10)
REMOVE_STOPWORDS = _env_bool("REMOVE_STOPWORDS", True)
LOWERCASE = _env_bool("LOWERCASE", True)
REMOVE_PUNCTUATION = _env_bool("REMOVE_PUNCTUATION", False)
SPACY_MODEL = _env_str("SPACY_MODEL", "en_core_web_sm")

# AWS configuration (used by the deployment docs / optional S3 helpers)
AWS_REGION = _env_str("AWS_REGION", "us-east-1")
AWS_BUCKET = _env_str("AWS_BUCKET", "search-relevancy-data")
