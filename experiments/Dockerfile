FROM python:3.11-slim

WORKDIR /app

# Install system deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 1) Copy packaging files first to leverage caching
COPY pyproject.toml ./
COPY monkeys/ ./monkeys/

# 2) Install monkeys package
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -e .

# 3) Copy only experiments code now
COPY experiments/src ./experiments/src

CMD ["python", "experiments/src/main.py"]
