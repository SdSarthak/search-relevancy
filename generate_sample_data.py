"""
Generate sample news articles dataset for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_sample_data(num_articles: int = 100) -> pd.DataFrame:
    """Generate sample news articles dataset."""
    
    categories = {
        'World': ['Politics', 'Conflict', 'Diplomacy'],
        'Technology': ['AI', 'Software', 'Hardware', 'Startups'],
        'Science': ['Environment', 'Medical', 'Space', 'Physics'],
        'Business': ['Markets', 'Companies', 'Economics', 'Finance'],
        'Entertainment': ['Movies', 'Music', 'Celebrity', 'Events']
    }
    
    sources = ['BBC', 'Reuters', 'CNN', 'Associated Press', 'The Guardian', 
               'New York Times', 'TechCrunch', 'Nature', 'Science Daily']
    
    sample_texts = {
        'climate': "Climate change continues to impact global weather patterns. New research shows rising temperatures affecting ecosystems worldwide.",
        'ai': "Artificial intelligence breakthroughs continue to transform industries. Machine learning models achieve new milestones in accuracy and efficiency.",
        'tech': "Technology sector shows strong growth with innovation in cloud computing, cybersecurity, and software development.",
        'business': "Global markets experience volatility as economic indicators fluctuate. Major corporations report quarterly earnings and strategic initiatives.",
        'health': "Medical research advances bring hope for new treatments. Healthcare systems worldwide adopt digital transformation strategies.",
        'politics': "Political developments reshape international relations. Government leaders engage in diplomatic discussions on pressing global issues.",
        'sports': "Athletes achieve remarkable performances in international competitions. Sports organizations announce new initiatives for sustainability.",
        'culture': "Cultural events celebrate diversity and creativity. Museums and galleries showcase diverse artistic perspectives and heritage.",
    }
    
    articles = []
    base_date = datetime.now() - timedelta(days=365)
    
    keywords = list(sample_texts.keys())
    
    for i in range(num_articles):
        category = np.random.choice(list(categories.keys()))
        subcategory = np.random.choice(categories[category])
        
        keyword = np.random.choice(keywords)
        base_text = sample_texts[keyword]
        
        article = {
            'article_id': f'article_{i+1:05d}',
            'category': category,
            'subcategory': subcategory,
            'title': f'{keyword.title()} News: Important Update {i+1}',
            'published_date': (base_date + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d'),
            'text': base_text + ' ' + ('Additional details and analysis provide deeper insights into the subject matter. ' * np.random.randint(1, 4)),
            'source': np.random.choice(sources)
        }
        articles.append(article)
    
    return pd.DataFrame(articles)


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample news articles dataset...")
    df = generate_sample_data(100)
    
    # Save to CSV
    output_path = Path(__file__).parent / "data" / "raw" / "news_articles.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    print(f"Generated {len(df)} sample articles")
    print("\nDataset preview:")
    print(df.head())
