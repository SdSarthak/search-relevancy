import numpy as np
import logging
from pathlib import Path
import pickle
from annoy import AnnoyIndex
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnnoyIndexBuilder:
    """Build and manage ANNOY approximate nearest neighbor index."""
    
    def __init__(self, embedding_dim: int, metric: str = "angular", num_trees: int = 10):
        """
        Initialize ANNOY index builder.
        
        Args:
            embedding_dim: Dimension of embeddings
            metric: Distance metric ('angular' for cosine similarity)
            num_trees: Number of trees for index (more = more accurate but slower)
        """
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.num_trees = num_trees
        self.index = AnnoyIndex(embedding_dim, metric=metric)
        logger.info(f"Initialized ANNOY index: dim={embedding_dim}, metric={metric}, trees={num_trees}")
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Numpy array of shape (num_items, embedding_dim)
        """
        logger.info(f"Adding {len(embeddings)} embeddings to index...")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        for idx, embedding in enumerate(embeddings):
            self.index.add_item(idx, embedding.tolist())
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Added {idx + 1} embeddings...")
        
        logger.info(f"Completed adding {len(embeddings)} embeddings")
    
    def build(self) -> None:
        """Build the index (this is required before searching)."""
        logger.info(f"Building index with {self.num_trees} trees...")
        self.index.build(self.num_trees)
        logger.info("Index building complete")
    
    def save(self, path: str) -> None:
        """
        Save index to file.
        
        Args:
            path: Path to save the index
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.index.save(path)
        logger.info(f"Index saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load index from file.
        
        Args:
            path: Path to load the index from
        """
        self.index.load(path)
        logger.info(f"Index loaded from {path}")
    
    def search(self, query_embedding: np.ndarray, num_results: int = 10) -> List[int]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query embedding vector
            num_results: Number of results to return
            
        Returns:
            List of indices of nearest neighbors
        """
        return self.index.get_nns_by_vector(query_embedding.tolist(), num_results)


def build_annoy_index(
    embeddings_path: str,
    metadata_path: str,
    index_output_path: str,
    num_trees: int = 10,
    metric: str = "angular"
) -> None:
    """
    Build ANNOY index from embeddings.
    
    Args:
        embeddings_path: Path to embeddings file (numpy array)
        metadata_path: Path to metadata file (pickle)
        index_output_path: Path to save the index
        num_trees: Number of trees for the index
        metric: Distance metric
    """
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    logger.info(f"Loaded metadata for {metadata['num_articles']} articles")
    
    # Verify dimensions match
    embedding_dim = metadata['embedding_dimension']
    if embeddings.shape[1] != embedding_dim:
        logger.error(f"Dimension mismatch: embeddings have {embeddings.shape[1]} dims, metadata expects {embedding_dim}")
        raise ValueError("Embedding dimension mismatch")
    
    # Build index
    builder = AnnoyIndexBuilder(
        embedding_dim=embedding_dim,
        metric=metric,
        num_trees=num_trees
    )
    
    builder.add_embeddings(embeddings)
    builder.build()
    builder.save(index_output_path)
    
    logger.info(f"ANNOY index successfully built and saved to {index_output_path}")


if __name__ == "__main__":
    from config.config import EMBEDDINGS_PATH, METADATA_PATH, ANNOY_INDEX_PATH, ANNOY_NUM_TREES, ANNOY_METRIC
    
    # Check if embeddings exist
    if not Path(EMBEDDINGS_PATH).exists():
        logger.warning(f"Embeddings not found at {EMBEDDINGS_PATH}")
        logger.info("Please run sbert_embeddings.py first")
    else:
        build_annoy_index(
            str(EMBEDDINGS_PATH),
            str(METADATA_PATH),
            str(ANNOY_INDEX_PATH),
            num_trees=ANNOY_NUM_TREES,
            metric=ANNOY_METRIC
        )
