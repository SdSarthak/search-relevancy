@echo off
REM Build the index from scratch and (optionally) start the API.
REM
REM   run_pipeline.bat            generate sample data, build, evaluate
REM   run_pipeline.bat --serve    ... then start the API
REM   set SKIP_SAMPLE=1           use the CSV already in data\raw

setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%PYTHON%"=="" set PYTHON=python

echo ======================================
echo Search Relevancy pipeline
echo ======================================

if "%SKIP_SAMPLE%"=="1" (
    echo.
    echo [1/5] skipping sample generation ^(SKIP_SAMPLE=1^)
) else (
    echo.
    echo [1/5] generating synthetic corpus
    %PYTHON% generate_sample_data.py || exit /b 1
)

echo.
echo [2/5] preprocessing articles
%PYTHON% -m src.data_preprocessing || exit /b 1

echo.
echo [3/5] generating SBERT embeddings
%PYTHON% -m src.sbert_embeddings || exit /b 1

echo.
echo [4/5] building the search index
%PYTHON% -m src.build_annoy_index || exit /b 1

echo.
echo [5/5] evaluating relevancy
%PYTHON% -m src.evaluate --sample-size 100 || exit /b 1

if "%1"=="--serve" (
    echo.
    echo starting the API on http://localhost:5000
    %PYTHON% -m src.app
    goto :eof
)

echo.
echo done. Query it with:  %PYTHON% -m src.search_cli "climate change"

endlocal
