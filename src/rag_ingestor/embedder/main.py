import logging
import json
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Depends,
    Query,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from functools import lru_cache
from rag_ingestor.embedder.service import EmbedderService
from rag_ingestor.embedder.models import EmbeddingRequest, EmbeddingResponse
from rag_ingestor.common.models import Document
from rag_ingestor.embedder.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Embedder Service",
    description="Service for creating document embeddings with support for large documents",
    version="1.0.0",
)


# Dependency for getting embedder service instance
@lru_cache
def get_embedder_service():
    return EmbedderService()


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    embedder_service: EmbedderService = Depends(get_embedder_service),
) -> EmbeddingResponse:
    """
    Create embeddings for a document.

    Args:
        request: Embedding request containing document and chunking parameters

    Returns:
        Response with generated embeddings
    """
    try:
        logger.info(f"Processing embedding request for document: {request.document.id}")
        return embedder_service.create_embeddings(request)
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class StreamEmbeddingRequest(BaseModel):
    document_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    chunk_size: Optional[int] = 512
    chunk_overlap: Optional[int] = 50


@app.post("/embed/stream")
async def create_embeddings_stream(
    request: StreamEmbeddingRequest,
    embedder_service: EmbedderService = Depends(get_embedder_service),
):
    """
    Stream embeddings for a large document.

    Args:
        request: Streaming embedding request

    Returns:
        Streaming response with generated embeddings
    """
    try:
        # Create document from request
        document = Document(
            id=request.document_id, content=request.content, metadata=request.metadata
        )

        # Stream embeddings asynchronously
        async def generate_stream():
            try:
                async for embedding in embedder_service.create_embeddings_stream(
                    document=document,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                ):
                    # Convert embedding to JSON and yield
                    embedding_json = json.dumps(embedding.dict()) + "\n"
                    yield embedding_json.encode("utf-8")
            except Exception as e:
                logger.error(f"Error in generate_stream: {str(e)}")
                yield json.dumps({"error": str(e)}).encode("utf-8")

        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="embeddings.jsonl"'},
        )
    except Exception as e:
        logger.error(f"Error in create_embeddings_stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/file")
async def create_embeddings_from_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = Query(None),
    metadata: Optional[str] = Query(None),
    chunk_size: int = Query(512),
    chunk_overlap: int = Query(50),
    embedder_service: EmbedderService = Depends(get_embedder_service),
):
    """
    Create embeddings from a file upload.

    Args:
        file: Uploaded file
        document_id: Optional document ID (will be generated if not provided)
        metadata: Optional metadata as JSON string
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        Streaming response with generated embeddings
    """
    try:
        # Generate document ID if not provided
        doc_id = document_id or f"doc_{file.filename}"

        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in metadata")

        # Add file information to metadata
        doc_metadata.update(
            {"filename": file.filename, "content_type": file.content_type}
        )

        # Function to read file content in chunks
        async def read_file_content():
            content_chunks = []
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                content_chunks.append(chunk)
            return b"".join(content_chunks)

        # Read the file content
        file_content = await read_file_content()

        # Try to decode the content as UTF-8
        try:
            text_content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            # Handle binary files or different encodings
            raise HTTPException(
                status_code=400,
                detail="File content could not be decoded as UTF-8. Only text files are supported.",
            )

        # Create document
        document = Document(id=doc_id, content=text_content, metadata=doc_metadata)

        # Stream embeddings asynchronously
        async def generate_stream():
            try:
                async for embedding in embedder_service.create_embeddings_stream(
                    document=document,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ):
                    # Convert embedding to JSON and yield
                    embedding_json = json.dumps(embedding.dict()) + "\n"
                    yield embedding_json.encode("utf-8")
            except Exception as e:
                logger.error(f"Error in generate_stream: {str(e)}")
                yield json.dumps({"error": str(e)}).encode("utf-8")

        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="embeddings.jsonl"'},
        )
    except Exception as e:
        logger.error(f"Error in create_embeddings_from_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health of the service"""
    return {"status": "healthy", "service": "embedder"}


@app.get("/info")
async def service_info(
    embedder_service: EmbedderService = Depends(get_embedder_service),
):
    """Get information about the embedder service"""
    try:
        # Get embedding model dimension
        dimension = embedder_service.embedding_model.dimension

        return {
            "service": "embedder",
            "embedding_model": settings.embedding_model,
            "embedding_dimension": dimension,
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Error in service_info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting embedder service on {settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port)
