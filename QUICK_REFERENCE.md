# 5-Minute Getting Started Guide

## Installation & Run (Choose One)

### Docker (Easiest)
```bash
cd "Search relevancy"
python generate_sample_data.py
docker-compose up --build
# API ready at http://localhost:5000
```

### Local Python
```bash
cd "Search relevancy"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python generate_sample_data.py
run_pipeline.bat
```

---

## Test It Works
```bash
# In another terminal
python test_api.py
```

---

## API Examples

### Search
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change", "num_results": 5}'
```

### Health Check
```bash
curl http://localhost:5000/health
```

### Python
```python
import requests
r = requests.post("http://localhost:5000/search", 
                  json={"query": "AI", "num_results": 5})
print(r.json())
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `src/app.py` | Flask API (endpoints: /search, /health, /info) |
| `src/data_preprocessing.py` | Clean & tokenize articles |
| `src/sbert_embeddings.py` | Create semantic embeddings |
| `src/build_annoy_index.py` | Build search index |
| `config/config.py` | Settings & paths |
| `Dockerfile` | Docker image |
| `docker-compose.yml` | Run everything |
| `test_api.py` | Test the API |
| `generate_sample_data.py` | Create demo data |

---

## Configuration

Edit `config/config.py`:
```python
SBERT_MODEL = "all-MiniLM-L6-v2"  # Change model
ANNOY_NUM_TREES = 10              # Accuracy vs speed
DEFAULT_NUM_RESULTS = 10          # Default results
```

---

## Troubleshooting

**Port 5000 in use?**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**Out of memory?**
- Reduce ANNOY_NUM_TREES in config.py
- Use lighter SBERT model

**Docker issues?**
```bash
docker-compose down
docker-compose up --build
```

---

## Deploy to AWS

See `AWS_DEPLOYMENT.md` for step-by-step instructions

---

## Key Endpoints

```
POST /search           - Search articles
POST /search/batch     - Multiple queries
GET  /health           - Check status
GET  /info             - Service info
```

---

## Documentation

- **Quick Start**: `QUICKSTART.md`
- **Full Docs**: `README.md`
- **AWS Deploy**: `AWS_DEPLOYMENT.md`
- **Project Info**: `PROJECT_BUILD_SUMMARY.md`

---

**You're ready to go!** 🚀
