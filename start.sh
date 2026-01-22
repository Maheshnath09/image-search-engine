#!/bin/bash
# Hugging Face Spaces Startup Script
# Runs both FastAPI backend and Streamlit frontend in a single container

echo "🚀 Starting Image Search Engine..."

# Create necessary directories
mkdir -p data/images data/coco data/cache vector_store

# Start FastAPI backend in background
echo "⚡ Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to initialize (CLIP model loading takes time)
echo "⏳ Waiting for backend to initialize..."
sleep 15

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy!"
else
    echo "⚠️ Backend may still be loading CLIP model..."
fi

# Start Streamlit frontend (this blocks and keeps container running)
echo "🖥️ Starting Streamlit frontend on port 8501..."
export API_URL="http://localhost:8000"
streamlit run frontend.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false
