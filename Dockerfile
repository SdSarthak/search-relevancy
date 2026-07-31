FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# build-essential is needed if you also install the optional annoy backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-annoy.txt ./

# The optional ANNOY backend is compiled here; if the build fails the image is
# still usable because the exact numpy backend needs no extension module.
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && (pip install -r requirements-annoy.txt || \
        echo "annoy unavailable, falling back to the exact index backend") \
    && python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY config/ ./config/
COPY generate_sample_data.py ./

# Artefacts are mounted at runtime (see docker-compose.yml); the directories
# must exist so the volume mount lands somewhere sensible.
RUN mkdir -p /app/models /app/data/raw /app/data/processed

EXPOSE 5000

ENV FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=0 \
    INDEX_BACKEND=auto \
    GUNICORN_WORKERS=2 \
    GUNICORN_TIMEOUT=120

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

# One worker per CPU is wasteful here: each holds its own copy of the model.
CMD ["sh", "-c", "gunicorn --workers ${GUNICORN_WORKERS} --timeout ${GUNICORN_TIMEOUT} --bind ${FLASK_HOST}:${FLASK_PORT} 'src.app:create_app()'"]
