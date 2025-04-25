# RAG Ingestor

A scalable RAG (Retrieval-Augmented Generation) application built with FastAPI, using llama-index for embeddings and Qdrant as the vector database. This project provides a modular and extensible solution for document ingestion, embedding generation, and vector storage.

## Project Structure

The project is organized into the following components:

```
rag-ingestor/
├── src/
│   └── rag_ingestor/
│       ├── common/         # Shared utilities and configurations
│       ├── embedder/       # Embedding generation service
│       └── indexer/        # Vector database management service
├── data/                   # Data directory for document storage
├── pyproject.toml          # Project dependencies and configuration
└── run.sh                  # Script to run the services
```

## Features

- **Modular Architecture**: Separate services for embedding generation and vector storage
- **FastAPI-based**: High-performance API endpoints with automatic OpenAPI documentation
- **Qdrant Integration**: Efficient vector similarity search
- **Configurable Embeddings**: Support for various embedding models through llama-index
- **Event System**: Integration with syft-event for monitoring and logging

## Prerequisites

- Python 3.9 or higher
- Docker (for running Qdrant)
- uv (Python package manager)

## Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd rag-ingestor
uv pip install -e .
```

3. Start Qdrant (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
EMBEDDER_HOST=0.0.0.0
EMBEDDER_PORT=8000
INDEXER_HOST=0.0.0.0
INDEXER_PORT=8001
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=6333
VECTOR_DB_COLLECTION=documents
VECTOR_DB_API_KEY=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384
```

## Running the Services

You can start all services using the provided `run.sh` script:

```bash
./run.sh
```

Or start individual services manually:

```bash
# Terminal 1 - Embedder Service
python -m src.rag_ingestor.embedder.main

# Terminal 2 - Indexer Service
python -m src.rag_ingestor.indexer.main
```

## API Endpoints

### Embedder Service (Port 8000)
- `POST /embed`: Create embeddings from a document
- `GET /health`: Health check endpoint

### Indexer Service (Port 8001)
- `POST /index`: Index embeddings in the vector database
- `POST /search`: Search for similar embeddings
- `GET /health`: Health check endpoint

## Usage Example

```python
import httpx

# Create embeddings
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/embed",
        json={
            "document": {
                "id": "doc1",
                "content": "Your document content here",
                "metadata": {}
            }
        }
    )

# Search for similar documents
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/search",
        json={
            "query": "Your search query here",
            "top_k": 5
        }
    )
```

## Development

### Setting Up Development Environment

1. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
```

2. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

### Adding New Dependencies

To add new dependencies, update the `dependencies` list in `pyproject.toml` and run:
```bash
uv pip install -e .
```

## Extensibility

The system is designed to be extensible:

1. **Vector Database**: Support for different vector databases can be added through the indexer service
2. **Embedding Models**: Various embedding models can be configured through the settings
3. **Event System**: Custom event handlers can be added through the syft-event integration

## License

MIT
