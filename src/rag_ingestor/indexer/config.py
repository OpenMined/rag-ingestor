from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class IndexerSettings(BaseSettings):
    """Settings for the indexer service."""

    # Service settings
    host: str = Field(default="0.0.0.0", description="Host for the indexer service")
    port: int = Field(default=8001, description="Port for the indexer service")

    # Vector database settings
    vector_db_host: str = Field(
        default="localhost", description="Host address of the vector database"
    )
    vector_db_port: int = Field(
        default=6333, description="Port number of the vector database"
    )
    vector_db_api_key: Optional[str] = Field(
        default=None, description="API key for the vector database (if needed)"
    )
    vector_db_collection: str = Field(
        default="embeddings",
        description="Name of the collection in the vector database",
    )

    # Embedding settings
    embedding_dimension: int = Field(
        default=384, description="Dimension of the embedding vectors"
    )
    similarity_metric: str = Field(
        default="Cosine",
        description="Similarity metric to use (Cosine, Dot, Euclidean)",
    )

    # Performance settings
    batch_size: int = Field(
        default=100, description="Batch size for indexing operations"
    )
    max_concurrent_requests: int = Field(
        default=5, description="Maximum number of concurrent requests"
    )
    qdrant_prefer_grpc: bool = Field(
        default=False, description="Whether to prefer gRPC over HTTP for Qdrant"
    )
    qdrant_timeout: float = Field(
        default=5.0, description="Connection timeout in seconds for Qdrant"
    )

    # Search settings
    default_top_k: int = Field(
        default=5, description="Default number of results to return from search"
    )

    class Config:
        env_prefix = "INDEXER_"
        env_file = ".env"


# Initialize settings instance
settings = IndexerSettings()


# Helper functions for working with different similarity metrics
def get_similarity_search_metric(metric: str) -> str:
    """
    Maps the storage similarity metric to the search metric.

    For example, if vectors are stored with cosine similarity,
    the search should use dot product with normalized vectors.

    Args:
        metric: The storage similarity metric

    Returns:
        The search similarity metric
    """
    mapping = {
        "Cosine": "dot_product",  # For cosine, use dot product with normalized vectors
        "Dot": "dot_product",  # For dot product, use dot product directly
        "Euclidean": "euclidean",  # For Euclidean, use Euclidean distance
    }
    return mapping.get(metric, "dot_product")
