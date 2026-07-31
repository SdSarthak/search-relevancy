import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SBERTEmbedder:
    """Generate SBERT embeddings for news articles."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SBERT embedder.
        
        Args:
            model_name: Name of the pre-trained SBERT model to use
        """
        logger.info(f"Loading SBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        logger.info(f"Encoding complete. Shape: {embeddings.shape}")
        return embeddings


def generate_embeddings(
    input_path: str,
    embeddings_output_path: str,
    metadata_output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> Tuple[np.ndarray, dict]:
    """
    Generate SBERT embeddings for processed articles.
    
    Args:
        input_path: Path to processed CSV file
        embeddings_output_path: Path to save embeddings
        metadata_output_path: Path to save metadata
        model_name: SBERT model name
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (embeddings array, metadata dict)
    """
    # Load processed data
    logger.info(f"Loading processed data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} articles")
    
    # Initialize embedder
    embedder = SBERTEmbedder(model_name=model_name)
    
    # Extract texts for embedding
    # Use processed_text if available, otherwise use title + text
    if 'processed_text' in df.columns:
        texts = df['processed_text'].tolist()
    else:
        texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()
    
    # Generate embeddings
    embeddings = embedder.encode(texts, batch_size=batch_size)
    
    # Prepare metadata
    metadata = {
        'article_ids': df['article_id'].tolist(),
        'titles': df['title'].tolist(),
        'categories': df['category'].tolist() if 'category' in df.columns else [],
        'subcategories': df['subcategory'].tolist() if 'subcategory' in df.columns else [],
        'sources': df['source'].tolist() if 'source' in df.columns else [],
        'published_dates': df['published_date'].tolist() if 'published_date' in df.columns else [],
        'texts': df['text'].tolist() if 'text' in df.columns else [],
        'embedding_model': model_name,
        'embedding_dimension': embedder.embedding_dim,
        'num_articles': len(df)
    }
    
    # Save embeddings
    Path(embeddings_output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_output_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_output_path}")
    
    # Save metadata
    Path(metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {metadata_output_path}")
    
    return embeddings, metadata


if __name__ == "__main__":
    from config.config import PROCESSED_DATA_PATH, EMBEDDINGS_PATH, METADATA_PATH, SBERT_MODEL
    
    # Check if processed data exists
    if not Path(PROCESSED_DATA_PATH).exists():
        logger.warning(f"Processed data not found at {PROCESSED_DATA_PATH}")
        logger.info("Please run data_preprocessing.py first")
    else:
        generate_embeddings(
            str(PROCESSED_DATA_PATH),
            str(EMBEDDINGS_PATH),
            str(METADATA_PATH),
            model_name=SBERT_MODEL
        )
