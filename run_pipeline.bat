@echo off
REM Complete pipeline runner script for Windows

setlocal enabledelayedexpansion

echo ======================================
echo Search Relevancy Pipeline
echo ======================================

REM Step 1: Generate sample data (optional)
echo.
echo [Step 1] Generating sample data...
python generate_sample_data.py
if !errorlevel! neq 0 (
    echo Error generating sample data
    exit /b 1
)

REM Step 2: Preprocess data
echo.
echo [Step 2] Preprocessing articles...
python src/data_preprocessing.py
if !errorlevel! neq 0 (
    echo Error preprocessing data
    exit /b 1
)

REM Step 3: Generate SBERT embeddings
echo.
echo [Step 3] Generating SBERT embeddings...
python src/sbert_embeddings.py
if !errorlevel! neq 0 (
    echo Error generating embeddings
    exit /b 1
)

REM Step 4: Build ANNOY index
echo.
echo [Step 4] Building ANNOY index...
python src/build_annoy_index.py
if !errorlevel! neq 0 (
    echo Error building ANNOY index
    exit /b 1
)

REM Step 5: Start Flask API
echo.
echo [Step 5] Starting Flask API...
echo API will be available at http://localhost:5000
python src/app.py

endlocal
