# Hugging Face Spaces Docker deployment
# Single container running both FastAPI backend and Streamlit frontend
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/images data/coco data/cache vector_store

# Make startup script executable
RUN chmod +x start.sh

# Expose ports (8000 for API, 7860 for Streamlit - HF Spaces requirement)
EXPOSE 8000 7860

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run startup script
CMD ["./start.sh"]