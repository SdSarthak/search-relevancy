#!/usr/bin/env bash
# Build the index from scratch and (optionally) start the API.
#
#   ./run_pipeline.sh              # generate sample data, build, evaluate
#   ./run_pipeline.sh --serve      # ... then start the API
#   SKIP_SAMPLE=1 ./run_pipeline.sh   # use the CSV already in data/raw

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
SERVE=0
[[ "${1:-}" == "--serve" ]] && SERVE=1

echo "======================================"
echo "Search Relevancy pipeline"
echo "======================================"

if [[ "${SKIP_SAMPLE:-0}" != "1" ]]; then
  echo
  echo "[1/5] generating synthetic corpus"
  "$PYTHON" generate_sample_data.py
else
  echo
  echo "[1/5] skipping sample generation (SKIP_SAMPLE=1)"
fi

echo
echo "[2/5] preprocessing articles"
"$PYTHON" -m src.data_preprocessing

echo
echo "[3/5] generating SBERT embeddings"
"$PYTHON" -m src.sbert_embeddings

echo
echo "[4/5] building the search index"
"$PYTHON" -m src.build_annoy_index

echo
echo "[5/5] evaluating relevancy"
"$PYTHON" -m src.evaluate --sample-size 100

if [[ "$SERVE" == "1" ]]; then
  echo
  echo "starting the API on http://localhost:${FLASK_PORT:-5000}"
  exec "$PYTHON" -m src.app
fi

echo
echo "done. Query it with:  $PYTHON -m src.search_cli \"climate change\""
