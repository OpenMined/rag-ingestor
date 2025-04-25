import logging
from functools import lru_cache
from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import Dict, Any
import time

from rag_ingestor.indexer.config import settings
from rag_ingestor.indexer.models import (
    IndexRequest,
    IndexResponse,
    DeleteRequest,
    DeleteResponse,
    DeleteByFilterRequest,
    SearchRequest,
    SearchResponse,
    IndexerStats,
)
from rag_ingestor.indexer.service import IndexerService


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Indexer Service",
    description="Service for indexing and searching embeddings in a vector database",
    version="1.0.0",
)


# Function to create the service with the appropriate configuration
@lru_cache
def create_indexer_service():
    """
    Create and configure the IndexerService.

    Returns:
        Configured IndexerService instance
    """
    # You can add custom configuration here if needed
    # Create config from settings
    config = {
        "vector_db_host": settings.vector_db_host,
        "vector_db_port": settings.vector_db_port,
        "vector_db_collection": settings.vector_db_collection,
        "embedding_dimension": settings.embedding_dimension,
        "similarity_metric": settings.similarity_metric,
        "batch_size": settings.batch_size,
        "default_top_k": settings.default_top_k,
    }

    # Use empty config to use settings from dedicated config module
    return IndexerService(config)


# Create a global service instance
indexer_service = create_indexer_service()


@app.post("/index", response_model=IndexResponse)
async def index_embeddings(request: IndexRequest) -> IndexResponse:
    """
    Index a batch of embeddings.

    Args:
        request: Request containing embeddings to index

    Returns:
        Response with indexing results
    """
    try:
        logger.info(f"Indexing {len(request.embeddings)} embeddings")
        result = indexer_service.index_embeddings(request.embeddings)
        return IndexResponse(**result)
    except Exception as e:
        logger.error(f"Error in index_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_embeddings(request: SearchRequest) -> SearchResponse:
    """
    Search for similar embeddings.

    Args:
        request: Search request

    Returns:
        Search results
    """
    try:
        start_time = time.time()

        # Use query as query_vector for backward compatibility
        results = indexer_service.search(
            query_vector=request.query,
            top_k=request.top_k,
            filter=request.filter if hasattr(request, "filter") else None,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Create response
        response = SearchResponse(
            results=results, total_found=len(results), duration_ms=duration_ms
        )

        return response
    except Exception as e:
        logger.error(f"Error in search_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete", response_model=DeleteResponse)
async def delete_embeddings(request: DeleteRequest) -> DeleteResponse:
    """
    Delete embeddings by IDs.

    Args:
        request: Request containing embedding IDs to delete

    Returns:
        Response with deletion results
    """
    try:
        logger.info(f"Deleting {len(request.ids)} embeddings")
        result = indexer_service.delete_embeddings(request.ids)
        return DeleteResponse(**result)
    except Exception as e:
        logger.error(f"Error in delete_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete/filter", response_model=DeleteResponse)
async def delete_by_filter(request: DeleteByFilterRequest) -> DeleteResponse:
    """
    Delete embeddings by filter.

    Args:
        request: Request containing filter for deletion

    Returns:
        Response with deletion results
    """
    try:
        logger.info(f"Deleting embeddings by filter")
        result = indexer_service.delete_by_filter(request.filter)
        return DeleteResponse(**result)
    except Exception as e:
        logger.error(f"Error in delete_by_filter: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_index")
async def batch_index_embeddings(
    request: IndexRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Index a large batch of embeddings asynchronously.

    Args:
        request: Request containing embeddings to index

    Returns:
        Initial response with task info
    """
    try:
        # Start background task
        logger.info(
            f"Starting background indexing of {len(request.embeddings)} embeddings"
        )

        # Add task to background tasks
        background_tasks.add_task(
            indexer_service.batch_index_embeddings, embeddings=request.embeddings
        )

        return {
            "success": True,
            "message": f"Batch indexing of {len(request.embeddings)} embeddings started",
            "status": "processing",
        }
    except Exception as e:
        logger.error(f"Error in batch_index_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=IndexerStats)
async def get_stats() -> IndexerStats:
    """
    Get statistics about the indexer service.

    Returns:
        Statistics about the indexer service
    """
    try:
        stats = indexer_service.get_stats()
        # Add database type if not present
        if "database_type" not in stats:
            stats["database_type"] = indexer_service.vector_db.__class__.__name__

        return IndexerStats(**stats)
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        return IndexerStats(total_embeddings=0, status="error", error=str(e))


@app.get("/health")
async def health_check():
    """
    Check the health of the service.

    Returns:
        Health status
    """
    return {"status": "healthy", "service": "indexer"}


@app.get("/config")
async def get_config():
    """
    Get the current configuration of the service.

    Returns:
        Current configuration values
    """
    # Return a subset of configuration values that are safe to expose
    return {
        "vector_db_host": settings.vector_db_host,
        "vector_db_port": settings.vector_db_port,
        "vector_db_collection": settings.vector_db_collection,
        "embedding_dimension": settings.embedding_dimension,
        "similarity_metric": settings.similarity_metric,
        "batch_size": settings.batch_size,
        "default_top_k": settings.default_top_k,
    }


if __name__ == "__main__":
    import uvicorn

    # Use settings from dedicated config
    host = settings.host
    port = settings.port

    logger.info(f"Starting indexer service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
