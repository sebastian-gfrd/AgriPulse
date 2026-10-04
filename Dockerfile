# AgriPulse: Ultra-Frugal Edge AI Engine Dockerfile for Cloud Run
FROM python:3.12-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORMS=cpu
ENV PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gTTS

# Copy application modules and precomputed artifacts
COPY agripulse/ agripulse/
COPY cli/ cli/
COPY data/ data/
COPY models/ models/
COPY audio/ audio/
COPY web/ web/
COPY app.py .

EXPOSE 8080

# Run lightweight edge server
CMD ["python3", "app.py"]
