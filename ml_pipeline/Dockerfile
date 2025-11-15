# ml_pipeline/Dockerfile
FROM python:3.11-slim

# Install build dependencies for scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model persistence directory
RUN mkdir -p /app/models/saved

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "pipeline.py"]
