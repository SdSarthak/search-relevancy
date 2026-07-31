# Search Relevancy - News Article Search Engine

A semantic search engine for news articles using Sentence-BERT (SBERT) embeddings and ANNOY approximate nearest neighbor indexing. The system is containerized with Docker and designed for deployment on AWS EC2 instances.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [AWS EC2 Deployment](#aws-ec2-deployment)

## Overview

This project implements a semantic search system for news articles that:

1. **Preprocesses** news article data (tokenization, lemmatization, stop word removal)
2. **Generates embeddings** using Sentence-BERT (SBERT) model for semantic representation
3. **Creates an index** using ANNOY for fast approximate nearest neighbor search
4. **Serves queries** through a Flask REST API with cosine similarity scoring

## Architecture

The system follows a three-phase architecture:

### Training Phase
```
Raw Data → Data Preprocessing → SBERT Embeddings → ANNOY Index
```

### Inference Phase
```
User Query → SBERT Encoding → ANNOY Search → Flask API → Results
```

### Deployment
```
Docker Container → AWS EC2 Instance → User Interface
```

## Tech Stack

- **Language**: Python 3.10
- **ML Models**: 
  - Sentence-BERT (all-MiniLM-L6-v2)
  - spaCy (NLP preprocessing)
- **Indexing**: ANNOY (Approximate Nearest Neighbors)
- **API Framework**: Flask with CORS
- **Containerization**: Docker & Docker Compose
- **Cloud**: AWS EC2
- **Data Processing**: pandas, numpy

## Project Structure

```
Search relevancy/
├── src/
│   ├── __init__.py
│   ├── app.py                    # Flask API application
│   ├── data_preprocessing.py      # Data preprocessing pipeline
│   ├── sbert_embeddings.py       # SBERT embedding generation
│   └── build_annoy_index.py      # ANNOY index building
├── config/
│   └── config.py                 # Configuration settings
├── data/
│   ├── raw/                      # Raw news articles CSV
│   └── processed/                # Processed articles
├── models/
│   ├── embeddings.npy            # SBERT embeddings
│   ├── metadata.pkl              # Article metadata
│   └── articles_index.annoy      # ANNOY index
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker Compose orchestration
├── requirements.txt              # Python dependencies
├── .dockerignore                 # Docker ignore patterns
├── .gitignore                    # Git ignore patterns
└── README.md                     # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerization)
- AWS account with EC2 access (for cloud deployment)
- At least 4GB RAM (for SBERT model loading)

### Local Development Setup

1. **Clone the repository**
   ```bash
   cd "c:\Users\sarth\OneDrive\Desktop\Projects\Search relevancy"
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Prepare data**
   - Place your CSV file at `data/raw/news_articles.csv`
   - Required columns: `article_id`, `category`, `subcategory`, `title`, `published_date`, `text`, `source`

5. **Run preprocessing pipeline**
   ```bash
   python src/data_preprocessing.py
   ```

6. **Generate SBERT embeddings**
   ```bash
   python src/sbert_embeddings.py
   ```

7. **Build ANNOY index**
   ```bash
   python src/build_annoy_index.py
   ```

8. **Start Flask API**
   ```bash
   python src/app.py
   ```

The API will be available at `http://localhost:5000`

## Usage

### Quick Start with Docker Compose

```bash
# Build and start the container
docker-compose up --build

# The API will be available at http://localhost:5000
```

### Data Format

The raw CSV file should have the following structure:

```csv
article_id,category,subcategory,title,published_date,text,source
1,World,Politics,Article Title,2023-01-15,"Full article text here...",BBC
2,Technology,AI,"Another Article",2023-01-16,"Article content...",TechCrunch
```

## API Endpoints

### 1. Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "Search Relevancy API",
  "num_articles": 22399
}
```

### 2. Search Articles
```http
POST /search
Content-Type: application/json

{
  "query": "climate change",
  "num_results": 10
}
```

Response:
```json
{
  "query": "climate change",
  "num_results": 2,
  "results": [
    {
      "article_id": "123",
      "title": "Global Climate Summit",
      "category": "Environment",
      "subcategory": "Climate",
      "source": "Reuters",
      "published_date": "2023-01-15",
      "text": "Full article text...",
      "relevance_score": 0.95
    },
    {
      "article_id": "456",
      "title": "Carbon Emissions Report",
      "category": "Science",
      "subcategory": "Environment",
      "source": "Nature",
      "published_date": "2023-01-14",
      "text": "Article text...",
      "relevance_score": 0.92
    }
  ]
}
```

### 3. Batch Search
```http
POST /search/batch
Content-Type: application/json

{
  "queries": ["climate change", "renewable energy"],
  "num_results": 5
}
```

### 4. Service Information
```http
GET /info
```

Response:
```json
{
  "service": "Search Relevancy API",
  "num_articles": 22399,
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "default_results": 10,
  "max_results": 50
}
```

## Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t search-relevancy:latest .

# Run the container
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  search-relevancy:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f search-api

# Stop services
docker-compose down
```

## AWS EC2 Deployment

### Step 1: Launch EC2 Instance

1. Go to AWS Management Console → EC2
2. Click "Launch Instance"
3. **AMI**: Ubuntu Server 22.04 LTS
4. **Instance Type**: t3.medium or larger (4GB RAM recommended)
5. **Security Group**: 
   - Allow SSH (port 22) from your IP
   - Allow HTTP (port 80) - optional, for load balancer
   - Allow port 5000 from your IP (or use load balancer)

### Step 2: Connect to Instance

```bash
ssh -i your-key.pem ubuntu@your-instance-public-ip
```

### Step 3: Install Docker & Docker Compose

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

### Step 4: Clone and Deploy Project

```bash
# Clone the repository
git clone https://github.com/your-username/search-relevancy.git
cd search-relevancy

# Create directories for models and data
mkdir -p models data/raw data/processed

# Copy your data to the instance (from your local machine)
scp -i your-key.pem data/raw/news_articles.csv ubuntu@your-instance:/home/ubuntu/search-relevancy/data/raw/

# Copy pre-generated models (if available)
scp -i your-key.pem -r models/* ubuntu@your-instance:/home/ubuntu/search-relevancy/models/
```

### Step 5: Build and Run with Docker Compose

```bash
# Navigate to project directory
cd search-relevancy

# Build and start containers
docker-compose up --build -d

# Check logs
docker-compose logs -f search-api

# Verify health
curl http://localhost:5000/health
```

### Step 6: Using AWS Application Load Balancer (Optional)

For production deployments, use AWS ALB:

1. **Create Target Group**
   - Port: 5000
   - Health check path: `/health`

2. **Create Application Load Balancer**
   - Listen on port 80
   - Route to target group

3. **Update Security Group**
   - Allow ALB to access EC2 instance on port 5000

### Step 7: Monitor and Maintain

```bash
# View running containers
docker-compose ps

# View detailed logs
docker-compose logs search-api

# Restart service
docker-compose restart search-api

# Update container
docker-compose down
docker-compose up --build -d
```

## Performance Optimization

### Configuration Tuning

Edit [config/config.py](config/config.py):

```python
# ANNOY tuning (more trees = more accurate but slower)
ANNOY_NUM_TREES = 10  # Increase for better accuracy

# Search results
DEFAULT_NUM_RESULTS = 10
MAX_NUM_RESULTS = 50

# Model selection (lighter models for faster inference)
SBERT_MODEL = "all-MiniLM-L6-v2"  # Fast, 384-dim
# Alternative: "all-mpnet-base-v2"  # Slower but more accurate, 768-dim
```

### EC2 Instance Sizing

| Dataset Size | Recommended Instance | RAM | vCPU |
|---|---|---|---|
| < 50K articles | t3.medium | 4 GB | 2 |
| 50K - 500K articles | t3.large | 8 GB | 2 |
| > 500K articles | t3.xlarge | 16 GB | 4 |

## Troubleshooting

### Model Loading Issues

```bash
# Download spacy model manually
python -m spacy download en_core_web_sm

# Verify SBERT model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Memory Issues on EC2

```bash
# Check available memory
free -h

# Monitor Docker usage
docker stats

# Limit container memory
docker run -m 2g ...
```

### Port Already in Use

```bash
# Check processes on port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

## API Usage Examples

### Python Client

```python
import requests

API_URL = "http://localhost:5000"

# Single search
response = requests.post(f"{API_URL}/search", json={
    "query": "artificial intelligence",
    "num_results": 5
})
results = response.json()
print(f"Found {results['num_results']} relevant articles")

for article in results['results']:
    print(f"- {article['title']} (relevance: {article['relevance_score']:.2%})")

# Get service info
info = requests.get(f"{API_URL}/info").json()
print(f"Total articles: {info['num_articles']}")
```

### cURL

```bash
# Search
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change", "num_results": 10}'

# Health check
curl http://localhost:5000/health

# Service info
curl http://localhost:5000/info
```

## Contributing

1. Create a feature branch
2. Make changes
3. Test locally with Docker Compose
4. Submit pull request

## License

MIT License - See LICENSE file for details

## Authors

Search Relevancy Team

## Support

For issues and questions:
- GitHub Issues
- Email: support@searchrelevancy.com

---

**Note**: This is a production-ready system. Always test thoroughly before deploying to production environments.
