import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re
from typing import Tuple
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spacy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


class NewsArticlePreprocessor:
    """Preprocess news articles for SBERT embedding."""
    
    def __init__(self, remove_stopwords=True, lowercase=True, remove_punctuation=True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits (keeping spaces)
        if self.remove_punctuation:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize and lemmatize text using spacy."""
        if not text or len(text) < 5:
            return ""
        
        doc = nlp(text[:1000000])  # Limit to first million chars for performance
        
        tokens = []
        for token in doc:
            # Skip stopwords if configured
            if self.remove_stopwords and token.is_stop:
                continue
            
            # Skip punctuation and spaces
            if token.is_punct or token.is_space:
                continue
            
            # Use lemma form
            tokens.append(token.lemma_)
        
        return ' '.join(tokens)
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        # Clean text
        text = self.clean_text(text)
        
        if len(text) < 10:
            return ""
        
        # Tokenize and lemmatize
        text = self.tokenize_and_lemmatize(text)
        
        return text


def preprocess_dataset(input_path: str, output_path: str, sample_size: int = None) -> None:
    """
    Load, preprocess, and save news articles dataset.
    
    Args:
        input_path: Path to raw CSV file
        output_path: Path to save processed CSV file
        sample_size: Optional sample size for testing
    """
    logger.info(f"Loading dataset from {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} articles")
    
    # Sample if specified
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logger.info(f"Using sample of {len(df)} articles")
    
    # Initialize preprocessor
    preprocessor = NewsArticlePreprocessor(
        remove_stopwords=True,
        lowercase=True,
        remove_punctuation=False  # Keep some punctuation
    )
    
    # Combine title and text for processing
    logger.info("Preprocessing articles...")
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Apply preprocessing
    df['processed_text'] = df['combined_text'].apply(
        lambda x: preprocessor.preprocess(x)
    )
    
    # Remove rows with empty processed text
    initial_count = len(df)
    df = df[df['processed_text'].str.len() > 0]
    logger.info(f"Removed {initial_count - len(df)} articles with empty processed text")
    
    # Keep only necessary columns
    output_df = df[['article_id', 'category', 'subcategory', 'title', 
                     'published_date', 'source', 'text', 'processed_text']].copy()
    
    # Save processed data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(output_df)} processed articles to {output_path}")
    
    return output_df


if __name__ == "__main__":
    from config.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
    
    # Check if raw data exists
    if not Path(RAW_DATA_PATH).exists():
        logger.warning(f"Raw data not found at {RAW_DATA_PATH}")
        logger.info("Please place your news_articles.csv in data/raw/")
    else:
        preprocess_dataset(str(RAW_DATA_PATH), str(PROCESSED_DATA_PATH))
