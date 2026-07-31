#!/bin/bash
# Complete pipeline runner script

set -e

echo "======================================"
echo "Search Relevancy Pipeline"
echo "======================================"

# Step 1: Generate sample data (optional)
echo ""
echo "[Step 1] Generating sample data..."
python generate_sample_data.py

# Step 2: Preprocess data
echo ""
echo "[Step 2] Preprocessing articles..."
python src/data_preprocessing.py

# Step 3: Generate SBERT embeddings
echo ""
echo "[Step 3] Generating SBERT embeddings..."
python src/sbert_embeddings.py

# Step 4: Build ANNOY index
echo ""
echo "[Step 4] Building ANNOY index..."
python src/build_annoy_index.py

# Step 5: Start Flask API
echo ""
echo "[Step 5] Starting Flask API..."
echo "API will be available at http://localhost:5000"
python src/app.py
