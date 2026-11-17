FROM python:3.11-slim

WORKDIR /app

# Install system audio libraries for live stream / file decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY aslytics ./aslytics
COPY web ./web
COPY tests ./tests

# Install core package + cross-platform ONNX runtime for Parakeet-TDT 0.6B v3
RUN pip install --no-cache-dir -e ".[onnx]"

EXPOSE 8765

CMD ["python3", "-m", "aslytics", "serve", "--host", "0.0.0.0", "--port", "8765"]
