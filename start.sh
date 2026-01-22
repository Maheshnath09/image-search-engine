#!/bin/bash
# Hugging Face Spaces Startup Script
# Runs indexer, FastAPI backend, and Streamlit frontend in a single container

echo "===== Application Startup at $(date) ====="
echo "🚀 Starting Image Search Engine..."

# Create necessary directories
mkdir -p data/images data/coco data/cache vector_store

# Check if COCO index already exists
if [ ! -f "vector_store/coco_index.bin" ]; then
    echo "📦 No COCO index found. Building index (this takes ~5-10 minutes on first run)..."
    echo "⏳ Downloading COCO dataset and generating embeddings..."
    python -c "
from indexer import build_coco_index
print('Starting COCO indexing...')
build_coco_index()
print('COCO indexing complete!')
"
    echo "✅ COCO index built successfully!"
else
    echo "✅ Found existing COCO index, skipping indexer..."
fi

# Start FastAPI backend in background
echo "⚡ Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to initialize
echo "⏳ Waiting for backend to initialize..."
sleep 15

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy!"
else
    echo "⚠️ Backend may still be loading CLIP model..."
fi

# Start Streamlit frontend (this blocks and keeps container running)
echo "🖥️ Starting Streamlit frontend on port 7860..."
export API_URL="http://localhost:8000"
streamlit run frontend.py
