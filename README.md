# Search Relevancy

Semantic search over a news corpus. Articles are cleaned with spaCy, embedded
with Sentence-BERT, indexed for nearest-neighbour retrieval and served through
a small Flask API — plus an evaluation harness that actually measures whether
the ranking is any good.

```
                     build time                          query time
 raw CSV ──► preprocess ──► SBERT embed ──► index    query ──► SBERT ──► index ──► ranked hits
             (spaCy)        (384-dim)      (annoy                                  (+ metadata)
                                           or exact)
```

## What is here

| Path | Purpose |
|---|---|
| `src/data_preprocessing.py` | Clean, lemmatise and de-noise raw articles |
| `src/sbert_embeddings.py` | Encode articles, write `embeddings.npy` + `metadata.pkl` |
| `src/vector_index.py` | Two interchangeable index backends (`annoy`, `exact`) |
| `src/build_annoy_index.py` | Pipeline step that builds and saves the index |
| `src/search_engine.py` | Encoder + index + metadata, framework-agnostic |
| `src/app.py` | Flask REST API (`create_app()` WSGI factory) |
| `src/search_cli.py` | Query the index from a terminal, no server needed |
| `src/evaluate.py` | Precision/Recall/MRR/nDCG and ANN-vs-exact recall |
| `config/config.py` | Every setting, overridable by environment variable |
| `tests/` | Offline pytest suite (no network, no model download) |
| `scripts/smoke_test_api.py` | Smoke test against a *running* API |

## Index backends

The project ships two backends behind one interface, both returning cosine
similarities in `[-1, 1]`:

| Backend | Install | Results | Use when |
|---|---|---|---|
| `exact` | included | exact | default; fine up to a few hundred thousand articles |
| `annoy` | `pip install -r requirements-annoy.txt` | approximate | large corpora, memory-mapped index |

`INDEX_BACKEND=auto` (the default) picks ANNOY when it is importable and falls
back to `exact` otherwise. ANNOY is a C++ extension without a wheel for every
platform — on Windows it needs the Microsoft C++ Build Tools — which is why it
is optional rather than a hard requirement.

Each backend writes its own file next to the configured index path:
`articles_index.annoy` or `articles_index.npz`.

## Setup

Requires Python 3.10+ and roughly 2 GB of RAM for the default MiniLM model.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

# optional: approximate index backend
pip install -r requirements-annoy.txt

cp .env.example .env            # optional; the defaults work as-is
```

## Getting data

No news corpus is bundled — the usual news datasets cannot be redistributed
here, and `data/` is gitignored.

**Synthetic corpus (works immediately).** Generates a topically-structured
fake corpus so the whole pipeline can be run and evaluated:

```bash
python generate_sample_data.py --num-articles 500
```

**Your own corpus.** Drop a CSV at `data/raw/news_articles.csv` with at least
`title` and `text`. `article_id`, `category`, `subcategory`, `published_date`
and `source` are used when present and backfilled when absent.

```csv
article_id,category,subcategory,title,published_date,text,source
1,World,Politics,Article title,2023-01-15,"Full article text...",BBC
```

Public corpora that fit this shape include the MIND news recommendation
dataset, the BBC News classification dataset and AG News; export them to the
columns above. Set `RAW_DATA_PATH` if you keep the file elsewhere.

## Running the pipeline

Either run the four steps, or use the wrapper:

```bash
./run_pipeline.sh          # Windows: run_pipeline.bat
./run_pipeline.sh --serve  # ... and start the API afterwards
```

```bash
python -m src.data_preprocessing     # data/raw -> data/processed
python -m src.sbert_embeddings       # -> models/embeddings.npy, models/metadata.pkl
python -m src.build_annoy_index      # -> models/articles_index.{annoy,npz}
python -m src.evaluate               # relevancy report
```

Every step takes `--help` and accepts explicit paths, so intermediate
artefacts can live anywhere.

## Querying

From the terminal:

```bash
python -m src.search_cli "ocean temperatures and glaciers" -n 5
python -m src.search_cli                 # interactive prompt
```

Or start the API:

```bash
python -m src.app                                            # dev server
gunicorn -w 2 -b 0.0.0.0:5000 "src.app:create_app()"         # production
```

## API

### `GET /health`

`200` once the index and model are loaded, `503` with the reason otherwise —
so it works as a container or load-balancer health check.

```json
{ "status": "healthy", "service": "Search Relevancy API", "num_articles": 500 }
```

### `POST /search`

```json
{ "query": "climate change", "num_results": 10 }
```

```json
{
  "query": "climate change",
  "num_results": 1,
  "results": [
    {
      "article_id": "article_00001",
      "title": "Climate: ocean temperatures reach new high",
      "category": "Science",
      "subcategory": "Environment",
      "source": "Reuters",
      "published_date": "2024-05-05",
      "text": "Ocean temperatures reach new high. Researchers report...",
      "relevance_score": 0.5699
    }
  ]
}
```

`relevance_score` is the cosine similarity between the query and article
embeddings: `1.0` identical, `0.0` unrelated, negative opposed. `text` is
truncated to `SNIPPET_CHARS` characters.

### `POST /search/batch`

```json
{ "queries": ["climate change", "renewable energy"], "num_results": 5 }
```

Up to 25 queries per request; responses come back under `batch_results`.

### `GET /info`

Article count, embedding model, embedding dimension and the active index
backend.

Errors are always JSON: `400` for a malformed request, `404` for an unknown
route, `405` for the wrong method, `503` when the index has not loaded, `500`
otherwise.

## Measuring relevancy

`src/evaluate.py` answers two separate questions.

**Is the ranking good?** Precision@k, Recall@k, MRR and nDCG@k against a
labelled query set. Supply your own:

```json
[{ "query": "climate change", "relevant_ids": ["article_00001"] }]
```

```bash
python -m src.evaluate --queries my_queries.json --k 1 5 10
```

Without one, the harness derives a *known-item* set: each article's title
becomes a query whose only relevant document is that article. It is a proxy,
not a substitute for human judgements, but it is reproducible and free.

**What does approximation cost?** The overlap between the live index's top-k
and exact brute-force top-k, alongside both latencies — the recall traded for
speed. With `INDEX_BACKEND=exact` the overlap is 1.0 by construction, which
makes it a useful regression guard either way.

```
$ python -m src.evaluate --sample-size 60
Search relevancy report
========================================
queries evaluated : 60
MRR               : 0.6311
  P@1    0.3833   R@1    0.3833   nDCG@1    0.3833
  P@5    0.2000   R@5    1.0000   nDCG@5    0.7243
  P@10   0.1000   R@10   1.0000   nDCG@10   0.7243
top-1 category match: 1.0000
latency ms        : mean 72.20 p50 11.70 p95 15.82
----------------------------------------
index backend     : exact
recall vs exact@10: 1.0000
latency ms        : approx 0.26 exact 0.24
```

Those numbers come from a 60-article *synthetic* corpus whose headlines repeat
across articles, which is why P@1 is low — they demonstrate the harness, not
the quality of a real corpus. Add `--report report.json` to persist metrics.

## Configuration

Everything lives in `config/config.py` and every value has an environment
override; see `.env.example`. The ones worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | `all-mpnet-base-v2` is slower but stronger (768-dim); changing it requires rebuilding the index |
| `INDEX_BACKEND` | `auto` | `auto`, `annoy` or `exact` |
| `ANNOY_NUM_TREES` | `10` | More trees: better recall, larger index |
| `DEFAULT_NUM_RESULTS` / `MAX_NUM_RESULTS` | `10` / `50` | Result count clamp |
| `SNIPPET_CHARS` | `400` | `0` returns the full stored body |
| `FLASK_DEBUG` | `0` | `1`/`true`/`yes`/`on` enable the debug server |
| `DATA_DIR` / `MODEL_DIR` | `./data`, `./models` | Keep artefacts outside the repo |

The dimension of a loaded index comes from `metadata.pkl`, so swapping
`SBERT_MODEL` without rebuilding fails loudly instead of returning nonsense.

## Docker

```bash
docker compose up --build      # API on http://localhost:5000
```

`models/` and `data/` are bind-mounted, so build the artefacts on the host
first (or exec into the container and run the pipeline). The image serves with
gunicorn and installs ANNOY opportunistically — the build succeeds either way.

For EC2 deployment see [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite is entirely offline: SBERT is replaced by a deterministic hashing
encoder and the corpus is a five-article fixture, so nothing downloads a model
or touches the network. The spaCy lemmatisation tests skip themselves when
`en_core_web_sm` is not installed.

To smoke test a running server:

```bash
python scripts/smoke_test_api.py --url http://localhost:5000
```

## Known limitations

- Retrieval is single-vector dense only. There is no lexical (BM25) leg and no
  hybrid fusion, so exact rare terms — product codes, surnames — can be missed.
- The index is rebuilt from scratch; there is no incremental add or delete.
- The evaluation query set is derived from titles unless labels are supplied.
  Real relevance judgements would be a considerably stronger signal.
- The API has no authentication or rate limiting; put it behind a gateway
  before exposing it publicly.

## License

MIT — see [LICENSE](LICENSE).
