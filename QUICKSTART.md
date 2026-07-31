# Quick Start Guide

Get the Search Relevancy application running in minutes!

## Prerequisites

- Python 3.10+ or Docker
- 4GB+ RAM recommended

---

## Option 1: Local Development (Python)

### 1. Setup Environment

```bash
cd "c:\Users\sarth\OneDrive\Desktop\Projects\Search relevancy"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Or (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Generate Sample Data

```bash
python generate_sample_data.py
# Creates sample news articles in data/raw/news_articles.csv
```

### 3. Run Full Pipeline

**Windows:**
```bash
run_pipeline.bat
```

**Mac/Linux:**
```bash
bash run_pipeline.sh
```

Or run steps individually:

```bash
# Preprocess data
python src/data_preprocessing.py

# Generate embeddings
python src/sbert_embeddings.py

# Build index
python src/build_annoy_index.py

# Start API
python src/app.py
```

### 4. Test the API

In a new terminal:
```bash
python test_api.py
```

Or manually:
```bash
# Health check
curl http://localhost:5000/health

# Search
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"climate change\", \"num_results\": 5}"
```

---

## Option 2: Docker (Recommended)

### 1. Prerequisites

- Docker installed
- Docker Compose installed

### 2. Generate Sample Data

```bash
cd "c:\Users\sarth\OneDrive\Desktop\Projects\Search relevancy"
python generate_sample_data.py
```

### 3. Run with Docker Compose

```bash
# Build and start
docker-compose up --build

# In another terminal, run pipeline or test
python test_api.py
```

Access the API at `http://localhost:5000`

### 4. Stop

```bash
docker-compose down
```

---

## API Examples

### Python

```python
import requests

# Search
response = requests.post("http://localhost:5000/search", json={
    "query": "climate change",
    "num_results": 5
})

results = response.json()
for article in results['results']:
    print(f"✓ {article['title']}")
    print(f"  Relevance: {article['relevance_score']:.1%}\n")
```

### cURL

```bash
# Single search
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"artificial intelligence","num_results":10}'

# Batch search
curl -X POST http://localhost:5000/search/batch \
  -H "Content-Type: application/json" \
  -d '{"queries":["AI","climate","tech"],"num_results":5}'

# Get info
curl http://localhost:5000/info

# Health check
curl http://localhost:5000/health
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/info` | GET | Service information |
| `/search` | POST | Search articles |
| `/search/batch` | POST | Batch search multiple queries |

### Search Request Format

```json
{
  "query": "your search term",
  "num_results": 10
}
```

### Search Response Format

```json
{
  "query": "your search term",
  "num_results": 2,
  "results": [
    {
      "article_id": "article_00001",
      "title": "Article Title",
      "category": "Category",
      "subcategory": "Subcategory",
      "source": "Source",
      "published_date": "2023-01-15",
      "text": "Article content...",
      "relevance_score": 0.95
    }
  ]
}
```

---

## Project Structure

```
Search relevancy/
├── src/                     # Source code
│   ├── app.py              # Flask API
│   ├── data_preprocessing.py
│   ├── sbert_embeddings.py
│   └── build_annoy_index.py
├── config/
│   └── config.py           # Configuration
├── data/
│   ├── raw/                # Raw data
│   └── processed/          # Processed data
├── models/                 # Models and indices
├── Dockerfile              # Docker image
├── docker-compose.yml      # Docker Compose config
├── requirements.txt        # Python dependencies
├── test_api.py             # API tests
└── README.md               # Full documentation
```

---

## Configuration

Edit `config/config.py` to customize:

```python
# SBERT model (trade-off between speed and accuracy)
SBERT_MODEL = "all-MiniLM-L6-v2"  # Fast, 384-dim
# or "all-mpnet-base-v2"  # More accurate, 768-dim

# ANNOY index accuracy (more trees = more accurate)
ANNOY_NUM_TREES = 10

# Search defaults
DEFAULT_NUM_RESULTS = 10
MAX_NUM_RESULTS = 50
```

---

## Troubleshooting

### Import errors
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Port already in use
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :5000
kill -9 <PID>
```

### Out of memory
```bash
# Reduce ANNOY trees or use lighter SBERT model
# See config.py
```

### Docker issues
```bash
# Rebuild
docker-compose down
docker-compose up --build

# Check logs
docker-compose logs search-api
```

---

## Next Steps

1. **Customize with your data**: Replace `news_articles.csv` in `data/raw/`
2. **Deploy to AWS**: See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)
3. **Integrate with frontend**: API ready at `http://localhost:5000`
4. **Production deployment**: Use Docker Compose on cloud

---

## Performance Tips

- **Faster inference**: Use lighter SBERT model
  - `all-MiniLM-L6-v2` (384-dim, fastest)
  - `all-mpnet-base-v2` (768-dim, more accurate)

- **Faster search**: Reduce ANNOY trees (less accurate)
  - `ANNOY_NUM_TREES = 5` (faster)
  - `ANNOY_NUM_TREES = 20` (more accurate)

- **Better results**: Add more training data

- **Scaling**: Use Docker Compose with load balancer

---

For detailed documentation, see [README.md](README.md) and [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)
