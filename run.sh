#!/bin/bash

# AI Knowledge Assistant - Run Script
# This script sets up environment variables and runs the Docker container

echo "🚀 Starting AI Knowledge Assistant..."

# Set OpenRouter API key
export OPENROUTER_API_KEY="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"
export OPENROUTER_MODEL="openai/gpt-oss-20b:free"

echo "✅ Environment variables set:"
echo "   - OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:0:20}..."
echo "   - OPENROUTER_MODEL: $OPENROUTER_MODEL"

# Create chroma_db directory if it doesn't exist
mkdir -p chroma_db

echo "📦 Building and starting Docker container..."

# Build and start the container
docker compose up -d --build

# Wait a moment for the container to start
echo "⏳ Waiting for container to start..."
sleep 5

# Check if the container is running
if docker compose ps | grep -q "Up"; then
    echo "✅ Container is running!"
    echo "🌐 API available at: http://localhost:8000"
    echo ""
    echo "📝 Quick test commands:"
    echo "   Upload documents:"
    echo "   curl -X POST \"http://localhost:8000/documents\" \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{\"documents\": [\"Machine learning is a subset of AI.\"]}'"
    echo ""
    echo "   Query the knowledge base:"
    echo "   curl \"http://localhost:8000/query?q=What%20is%20machine%20learning?\""
    echo ""
    echo "📊 View logs: docker compose logs -f api"
    echo "🛑 Stop container: docker compose down"
else
    echo "❌ Container failed to start. Check logs with: docker compose logs api"
    exit 1
fi

