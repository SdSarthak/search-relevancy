import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw" / "news_articles.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_articles.csv"
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.npy"
METADATA_PATH = MODEL_DIR / "metadata.pkl"

# SBERT Configuration
SBERT_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast model
EMBEDDING_DIMENSION = 384

# ANNOY Configuration
ANNOY_INDEX_PATH = MODEL_DIR / "articles_index.annoy"
ANNOY_NUM_TREES = 10
ANNOY_METRIC = "angular"  # cosine similarity

# Flask Configuration
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", False)

# Search Configuration
DEFAULT_NUM_RESULTS = 10
MAX_NUM_RESULTS = 50

# Preprocessing Configuration
MIN_TEXT_LENGTH = 10
REMOVE_STOPWORDS = True
LOWERCASE = True
REMOVE_PUNCTUATION = True

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET = os.getenv("AWS_BUCKET", "search-relevancy-data")
