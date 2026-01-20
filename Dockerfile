# Brain Tumor CDSS Docker Image
# Multi-stage build for smaller image size

# Stage 1: Base image with dependencies
FROM python:3.9-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM base as production

WORKDIR /app

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Create directories for weights and data
RUN mkdir -p /app/weights /app/data

# Default command
CMD ["python", "-c", "print('Brain Tumor CDSS ready. Use: python inference/predict.py --help')"]

# Stage 3: Development image with additional tools
FROM base as development

WORKDIR /app

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    jupyter \
    jupyterlab

COPY . .

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 4: API server image
FROM base as api

WORKDIR /app

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart

COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
