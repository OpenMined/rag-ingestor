from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union
from uuid import UUID

from rag_ingestor.common.models import Embedding


class SearchFilter(BaseModel):
    """
    Filter for vector database searches.
    """

    exact_filters: Dict[str, Any] = Field(
        default_factory=dict, description="Exact match filters (field: value)"
    )
    range_filters: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Range filters (field: {'gt': value, 'lt': value, 'gte': value, 'lte': value})",
    )


class IndexRequest(BaseModel):
    """Request model for indexing embeddings."""

    embeddings: List[Embedding] = Field(..., description="List of embeddings to index")


class IndexResponse(BaseModel):
    """Response model for indexing operation."""

    success: bool = Field(..., description="Whether the operation was successful")
    indexed_count: int = Field(..., description="Number of embeddings indexed")
    duration_ms: Optional[int] = Field(
        None, description="Duration of the operation in milliseconds"
    )
    error: Optional[str] = Field(
        None, description="Error message if the operation failed"
    )


class DeleteRequest(BaseModel):
    """Request model for deleting embeddings by IDs."""

    ids: List[UUID] = Field(..., description="List of embedding IDs to delete")

    @field_validator("ids", mode="before")
    @classmethod
    def validate_ids(cls, v):
        """Convert string IDs to UUID if possible"""
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    try:
                        result.append(UUID(item))
                    except ValueError:
                        # Keep as string if not a valid UUID
                        result.append(item)
                else:
                    result.append(item)
            return result
        return v


class DeleteByFilterRequest(BaseModel):
    """Request model for deleting embeddings by filter."""

    filter: Union[Dict[str, Any], SearchFilter] = Field(
        ..., description="Filter to apply for deletion"
    )

    @field_validator("filter", mode="before")
    @classmethod
    def validate_filter(cls, v):
        """Convert dict filter to SearchFilter"""
        if isinstance(v, dict):
            return SearchFilter(exact_filters=v)
        return v


class DeleteResponse(BaseModel):
    """Response model for deletion operation."""

    success: bool = Field(..., description="Whether the operation was successful")
    deleted_count: int = Field(..., description="Number of embeddings deleted")
    duration_ms: Optional[int] = Field(
        None, description="Duration of the operation in milliseconds"
    )
    error: Optional[str] = Field(
        None, description="Error message if the operation failed"
    )


class SearchRequest(BaseModel):
    """Request model for searching embeddings."""

    query: List[float] = Field(..., description="Query embedding vector")
    top_k: int = Field(default=5, description="Number of results to return")
    filter: Optional[Union[Dict[str, Any], SearchFilter]] = Field(
        default=None, description="Filter to apply to the search"
    )

    @field_validator("filter", mode="before")
    @classmethod
    def validate_filter(cls, v):
        """Convert dict filter to SearchFilter if provided"""
        if v is not None and isinstance(v, dict):
            return SearchFilter(exact_filters=v)
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        """Validate top_k is positive"""
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v


class SearchResult(BaseModel):
    """Model for a single search result."""

    id: UUID = Field(..., description="ID of the matching embedding")
    document_id: UUID = Field(..., description="ID of the original document")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata of the embedding"
    )
    content: Optional[str] = Field(
        None, description="Content of the matched chunk/document"
    )

    @field_validator("id", "document_id", mode="before")
    @classmethod
    def validate_ids(cls, v):
        """Convert string IDs to UUID if possible"""
        if isinstance(v, str):
            try:
                return UUID(v)
            except ValueError:
                # Keep as string if not a valid UUID
                return v
        return v

    @field_validator("document_id", mode="before")
    @classmethod
    def validate_document_id(cls, v):
        """Convert string IDs to UUID if possible"""
        if isinstance(v, str):
            return UUID(v)
        return v


class SearchResponse(BaseModel):
    """Response model for search operation."""

    results: List[SearchResult] = Field(..., description="Search results")
    total_found: Optional[int] = Field(
        None, description="Total number of results found"
    )
    duration_ms: Optional[int] = Field(
        None, description="Duration of the search in milliseconds"
    )


class IndexerStats(BaseModel):
    """Statistics about the indexer service."""

    total_embeddings: int = Field(
        ..., description="Total number of embeddings in the database"
    )
    status: str = Field(..., description="Status of the indexer service")
    error: Optional[str] = Field(None, description="Error message if any")
    collection_name: Optional[str] = Field(
        None, description="Name of the collection in the vector database"
    )
    database_type: Optional[str] = Field(
        None, description="Type of the vector database"
    )
