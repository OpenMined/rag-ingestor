from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from uuid import uuid4, UUID


class Document(BaseModel):
    """
    Represents a document to be processed by the RAG system.
    """

    id: UUID = Field(
        ...,
        description="Unique identifier for the document",
        default_factory=lambda: uuid4(),
    )
    content: str = Field(..., description="Content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the document"
    )

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, v):
        """Validate that id is a valid UUID"""
        if isinstance(v, str):
            try:
                return UUID(v)
            except ValueError:
                raise ValueError("id must be a valid UUID")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        """Validate that content is not empty"""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v


class Embedding(BaseModel):
    """
    Represents a vector embedding of a document or document chunk.
    """

    id: UUID = Field(
        ...,
        description="Unique identifier for the embedding",
        default_factory=lambda: uuid4(),
    )
    vector: List[float] = Field(
        ..., description="Vector representation of the document"
    )
    document_id: UUID = Field(..., description="Reference to the original document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the embedding"
    )

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, v):
        """Validate that id is a valid UUID"""
        if isinstance(v, str):
            try:
                return UUID(v)
            except ValueError:
                raise ValueError("id must be a valid UUID")
        return v

    @field_validator("vector", mode="before")
    @classmethod
    def validate_vector(cls, v):
        """Validate that vector is not empty"""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        return v
