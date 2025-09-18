# Slim, fast, secure base
FROM python:3.11-slim

# Good defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# BLAS/OpenMP runtime & build basics
# libgomp1: OpenMP runtime needed by scikit-learn wheels
# curl/git: handy for experiments; remove if you don't need them
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for nicer permissions
RUN useradd -m -u 1000 appuser
WORKDIR /app
USER appuser

# Leverage layer caching for deps
COPY --chown=appuser:appuser requirements.txt .
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

# Copy source last for faster rebuilds while iterating
COPY --chown=appuser:appuser src/ ./src

# Threading knobs (tune at runtime if you like)
# These cap BLAS/OpenMP threads to avoid oversubscription.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

CMD ["python", "src/hello.py"]

