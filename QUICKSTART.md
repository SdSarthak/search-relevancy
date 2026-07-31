# Quick start

From a clean checkout to a working search API. Roughly five minutes, most of
it spent downloading the SBERT model the first time.

## 1. Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional, for the approximate index backend (needs a C++ toolchain):

```bash
pip install -r requirements-annoy.txt
```

Without it the project uses the built-in exact numpy index — everything works,
results are exact, and it is fast enough well past 100k articles.

## 2. Build an index

```bash
./run_pipeline.sh                 # Windows: run_pipeline.bat
```

That generates a synthetic corpus, preprocesses it, embeds it, builds the
index and prints a relevancy report. To use your own data instead, put a CSV
with `title` and `text` columns at `data/raw/news_articles.csv` and run:

```bash
SKIP_SAMPLE=1 ./run_pipeline.sh   # Windows: set SKIP_SAMPLE=1 && run_pipeline.bat
```

Artefacts land in `models/`:

```
models/embeddings.npy          the encoded corpus
models/metadata.pkl            index position -> article
models/articles_index.npz      exact backend  (or .annoy for ANNOY)
```

## 3. Search

Terminal:

```bash
python -m src.search_cli "ocean temperatures and glaciers" -n 5
```

API:

```bash
python -m src.app
```

```bash
curl http://localhost:5000/health

curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "climate change", "num_results": 5}'
```

Python client:

```python
import requests

response = requests.post(
    "http://localhost:5000/search",
    json={"query": "artificial intelligence", "num_results": 5},
)
for hit in response.json()["results"]:
    print(f"{hit['relevance_score']:+.4f}  {hit['title']}")
```

## 4. Check the quality

```bash
python -m src.evaluate --sample-size 200 --k 1 5 10
```

Reports Precision/Recall/nDCG@k, MRR, query latency and — when ANNOY is in use
— how much recall the approximation costs versus exact search.

## Docker

```bash
docker compose up --build
```

`models/` and `data/` are bind-mounted, so build the artefacts on the host
first (step 2).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Offline: no model download, no network, no server.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `spaCy model 'en_core_web_sm' is not installed` | `python -m spacy download en_core_web_sm` |
| `no index found (looked for: ...)` | Run the pipeline (step 2) |
| `metadata not found` | Run `python -m src.sbert_embeddings` |
| `/health` returns 503 | Read `details` in the response; the artefacts are usually missing or the volume is not mounted |
| `pip install annoy` fails on Windows | Skip it — the exact backend is the default fallback |
| Port 5000 already in use | `FLASK_PORT=5001 python -m src.app` |
| Model download blocked | Pre-download on a connected machine and copy `~/.cache/huggingface` across |

More detail in [README.md](README.md); EC2 deployment in
[AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).
