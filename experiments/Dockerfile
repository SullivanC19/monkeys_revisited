# experiments/Dockerfile
FROM python:3.11-slim

# OpenMP runtime for scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy just what’s needed to resolve/install deps (maximizes build cache)
COPY pyproject.toml ./
COPY monkeys/ ./monkeys/

# Install your package + selected extras directly from pyproject
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -e .[viz,io,hf]

# Bring in experiment code last so edits don't bust the deps cache
COPY experiments/src ./experiments/src

CMD ["python", "experiments/src/main.py"]
