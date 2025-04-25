from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from rag_ingestor.common.models import Document, Embedding
from enum import Enum


class ChunkingStrategy(str, Enum):
    """Enum for chunking strategies."""

    SIMPLE = "simple"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class EmbeddingRequest(BaseModel):
    """Request model for creating embeddings."""

    document: Document
    chunk_size: Optional[int] = Field(
        default=512, description="Size of text chunks for processing", ge=1
    )
    chunk_overlap: Optional[int] = Field(
        default=50, description="Overlap between chunks", ge=0
    )
    chunking_strategy: Optional[ChunkingStrategy] = Field(
        default=ChunkingStrategy.SIMPLE, description="Strategy to use for chunking text"
    )
    batch_size: Optional[int] = Field(
        default=10, description="Batch size for processing chunks", ge=1
    )

    @field_validator("chunk_overlap")
    def validate_overlap(cls, v, values):
        """Validate that overlap is less than chunk size."""
        if "chunk_size" in values and v is not None and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class EmbeddingResponse(BaseModel):
    """Response model for embedding creation."""

    embeddings: List[Embedding]
    total_chunks: int = Field(..., description="Total number of chunks processed")
    model_name: Optional[str] = Field(
        None, description="Name of the embedding model used"
    )
    model_dimension: Optional[int] = Field(
        None, description="Dimension of the embedding vectors"
    )

    class Config:
        schema_extra = {
            "example": {
                "embeddings": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "vector": [0.1, 0.2, 0.3, 0.4],
                        "document_id": "doc_123",
                        "metadata": {
                            "chunk_text": "This is a sample chunk of text.",
                            "chunk_index": 0,
                        },
                    }
                ],
                "total_chunks": 1,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_dimension": 384,
            }
        }


class EmbeddingStats(BaseModel):
    """Statistics about the embedding process."""

    document_id: str
    total_chunks: int
    total_tokens: Optional[int] = None
    processing_time_ms: float
    avg_chunk_size: float
    embedding_dimension: int


class EmbeddingStreamRequest(BaseModel):
    """Request model for streaming embeddings."""

    document_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    chunk_size: Optional[int] = Field(default=512, ge=1)
    chunk_overlap: Optional[int] = Field(default=50, ge=0)
    chunking_strategy: Optional[ChunkingStrategy] = Field(
        default=ChunkingStrategy.SIMPLE
    )

    @field_validator("chunk_overlap")
    def validate_overlap(cls, v, values):
        """Validate that overlap is less than chunk size."""
        if "chunk_size" in values and v is not None and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class EmbeddingBatchRequest(BaseModel):
    """Request model for batch embedding."""

    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None

    @field_validator("metadata")
    def validate_metadata_length(cls, v, values):
        """Validate that metadata length matches texts length if provided."""
        if v is not None and "texts" in values and len(v) != len(values["texts"]):
            raise ValueError("metadata length must match texts length")
        return v


class EmbeddingBatchResponse(BaseModel):
    """Response model for batch embedding."""

    embeddings: List[List[float]]
    model_name: str
    model_dimension: int
