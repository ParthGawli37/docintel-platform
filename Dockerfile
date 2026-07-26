# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# tesseract-ocr: required by the default OCRProvider (TesseractOCRProvider)
# libmagic1: used transitively by some document-parsing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data/raw /app/data/cache

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "docintel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
