#!/bin/bash

set -e

# Create and activate virtual environment
uv venv -p 3.12 .venv1
. .venv1/bin/activate
uv pip install -e .

# Configuration
EMBEDDER_PORT=8000
INDEXER_PORT=8001
QDRAINT_PORT=6333

# Function to check if a port is in use
check_port() {
    local port=$1
    local service=$2
    if lsof -i :$port > /dev/null 2>&1; then
        echo "Error: Port $port is already in use. Please make sure $service is not running."
        exit 1
    fi
}

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    if [ ! -z "$EMBEDDER_PID" ]; then
        kill $EMBEDDER_PID 2>/dev/null || true
    fi
    if [ ! -z "$INDEXER_PID" ]; then
        kill $INDEXER_PID 2>/dev/null || true
    fi
    if [ ! -z "$QDRAINT_PID" ]; then
        docker stop $QDRAINT_CONTAINER_ID 2>/dev/null || true
    fi
    exit 0
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local service=$2
    local max_attempts=30
    local attempt=1

    echo "Waiting for $service to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url/health" > /dev/null; then
            echo "$service is ready!"
            return 0
        fi
        echo "Attempt $attempt: $service not ready yet, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo "Error: $service failed to start within the expected time"
    cleanup
    return 1
}

# Set up signal handlers
trap 'cleanup' 2 15  # 2 is SIGINT, 15 is SIGTERM

# Check if required ports are available
check_port $EMBEDDER_PORT "Embedder"
check_port $INDEXER_PORT "Indexer"
check_port $QDRAINT_PORT "Qdrant"

# Start Qdrant database
echo "Starting Qdrant database..."
docker run -d -p $QDRAINT_PORT:$QDRAINT_PORT qdrant/qdrant
QDRAINT_CONTAINER_ID=$(docker ps -q --filter ancestor=qdrant/qdrant)
QDRAINT_PID=$QDRAINT_CONTAINER_ID

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
sleep 5  # Give Qdrant some time to start

# Start embedder service
echo "Starting embedder service..."
python3 -m rag_ingestor.embedder.main &
EMBEDDER_PID=$!

# Start indexer service
echo "Starting indexer service..."
python3 -m rag_ingestor.indexer.main &
INDEXER_PID=$!

# Wait for services to be ready
wait_for_service "http://localhost:$EMBEDDER_PORT" "Embedder"
wait_for_service "http://localhost:$INDEXER_PORT" "Indexer"

# Wait for all processes to finish
wait $EMBEDDER_PID $INDEXER_PID
