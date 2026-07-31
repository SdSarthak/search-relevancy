import pickle
import numpy as np
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    SBERT_MODEL, EMBEDDING_DIMENSION, ANNOY_INDEX_PATH, METADATA_PATH,
    DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS, FLASK_HOST, FLASK_PORT, FLASK_DEBUG
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for model and index
sbert_model = None
annoy_index = None
metadata = None


def initialize_models():
    """Load SBERT model, ANNOY index, and metadata."""
    global sbert_model, annoy_index, metadata
    
    logger.info("Initializing models...")
    
    # Load SBERT model
    try:
        logger.info(f"Loading SBERT model: {SBERT_MODEL}")
        sbert_model = SentenceTransformer(SBERT_MODEL)
        logger.info("SBERT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load SBERT model: {str(e)}")
        raise
    
    # Load metadata
    try:
        logger.info(f"Loading metadata from {METADATA_PATH}")
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"Metadata loaded: {metadata['num_articles']} articles")
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        raise
    
    # Load ANNOY index
    try:
        logger.info(f"Loading ANNOY index from {ANNOY_INDEX_PATH}")
        annoy_index = AnnoyIndex(EMBEDDING_DIMENSION, metric='angular')
        annoy_index.load(str(ANNOY_INDEX_PATH))
        logger.info("ANNOY index loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ANNOY index: {str(e)}")
        raise
    
    logger.info("All models initialized successfully")


def encode_query(query: str) -> np.ndarray:
    """
    Encode query string to embedding using SBERT.
    
    Args:
        query: Query text
        
    Returns:
        Embedding vector
    """
    embedding = sbert_model.encode(query, convert_to_numpy=True)
    return embedding


def search_articles(query: str, num_results: int = DEFAULT_NUM_RESULTS) -> Dict[str, Any]:
    """
    Search for relevant articles using query.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Dictionary containing search results
    """
    # Validate num_results
    num_results = max(1, min(num_results, MAX_NUM_RESULTS))
    
    # Encode query
    logger.info(f"Encoding query: '{query}'")
    query_embedding = encode_query(query)
    
    # Search in ANNOY index
    logger.info(f"Searching for {num_results} nearest neighbors...")
    article_indices = annoy_index.get_nns_by_vector(
        query_embedding.tolist(),
        num_results,
        include_distances=True
    )
    
    # Build results
    results = []
    for idx, distance in zip(article_indices[0], article_indices[1]):
        # Convert distance to similarity (0-1 range)
        # For angular distance, similarity = 1 - distance/pi
        similarity = 1 - (distance / np.pi)
        
        result = {
            'article_id': metadata['article_ids'][idx],
            'title': metadata['titles'][idx],
            'category': metadata['categories'][idx] if metadata['categories'] else None,
            'subcategory': metadata['subcategories'][idx] if metadata['subcategories'] else None,
            'source': metadata['sources'][idx] if metadata['sources'] else None,
            'published_date': metadata['published_dates'][idx] if metadata['published_dates'] else None,
            'text': metadata['texts'][idx] if metadata['texts'] else None,
            'relevance_score': float(similarity)
        }
        results.append(result)
    
    return {
        'query': query,
        'num_results': len(results),
        'results': results
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Search Relevancy API',
        'num_articles': metadata['num_articles'] if metadata else 0
    }), 200


@app.route('/search', methods=['POST'])
def search():
    """
    Search endpoint.
    
    Request body:
        {
            "query": "search query string",
            "num_results": 10 (optional)
        }
    
    Response:
        {
            "query": "search query string",
            "num_results": 10,
            "results": [
                {
                    "article_id": "...",
                    "title": "...",
                    "category": "...",
                    "subcategory": "...",
                    "source": "...",
                    "published_date": "...",
                    "text": "...",
                    "relevance_score": 0.95
                }
            ]
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        query = data['query'].strip()
        if not query or len(query) < 2:
            return jsonify({'error': 'Query must be at least 2 characters long'}), 400
        
        num_results = data.get('num_results', DEFAULT_NUM_RESULTS)
        
        # Perform search
        logger.info(f"Processing search request: query='{query}', num_results={num_results}")
        results = search_articles(query, num_results)
        
        return jsonify(results), 200
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/search/batch', methods=['POST'])
def batch_search():
    """
    Batch search endpoint for multiple queries.
    
    Request body:
        {
            "queries": ["query1", "query2"],
            "num_results": 10 (optional)
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({'error': 'Missing required field: queries'}), 400
        
        queries = data['queries']
        if not isinstance(queries, list) or len(queries) == 0:
            return jsonify({'error': 'queries must be a non-empty list'}), 400
        
        num_results = data.get('num_results', DEFAULT_NUM_RESULTS)
        
        # Process queries
        batch_results = []
        for query in queries:
            query = query.strip()
            if query and len(query) >= 2:
                results = search_articles(query, num_results)
                batch_results.append(results)
        
        return jsonify({'batch_results': batch_results}), 200
    
    except Exception as e:
        logger.error(f"Batch search error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/info', methods=['GET'])
def info():
    """Get information about the search service."""
    if not metadata:
        return jsonify({'error': 'Service not initialized'}), 503
    
    return jsonify({
        'service': 'Search Relevancy API',
        'num_articles': metadata['num_articles'],
        'embedding_model': metadata['embedding_model'],
        'embedding_dimension': metadata['embedding_dimension'],
        'default_results': DEFAULT_NUM_RESULTS,
        'max_results': MAX_NUM_RESULTS
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize models before starting server
    initialize_models()
    
    # Start Flask app
    logger.info(f"Starting Flask server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
