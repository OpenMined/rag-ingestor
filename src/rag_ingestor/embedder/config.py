from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Dict, Any, List


class EmbedderSettings(BaseSettings):
    """Settings for the embedder service."""

    # Service settings
    host: str = Field(default="0.0.0.0", description="Host for the embedder service")
    port: int = Field(default=8000, description="Port for the embedder service")

    # Application info
    app_name: str = Field(
        default="RAG Embedder Service", description="Name of the service"
    )
    app_version: str = Field(default="1.0.0", description="Version of the service")

    # Embedding model settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the HuggingFace embedding model",
    )

    # Default chunking settings
    default_chunk_size: int = Field(
        default=512, description="Default size of text chunks for processing"
    )
    default_chunk_overlap: int = Field(
        default=50, description="Default overlap between chunks"
    )

    # Performance settings
    batch_size: int = Field(default=10, description="Batch size for processing chunks")
    max_concurrent_requests: int = Field(
        default=5, description="Maximum number of concurrent embedding requests"
    )

    # Resource limits
    max_document_size_mb: float = Field(
        default=10.0, description="Maximum document size in MB"
    )

    # Logging settings
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )

    # CORS settings
    cors_origins: List[str] = Field(
        default=["*"], description="List of allowed CORS origins"
    )

    # Security settings
    api_key_enabled: bool = Field(
        default=False, description="Whether to enable API key authentication"
    )

    class Config:
        env_prefix = "EMBEDDER_"
        env_file = ".env"


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific embedding model.

    Args:
        model_name: Name of the embedding model

    Returns:
        Configuration for the model
    """
    # Default configurations for common models
    model_configs = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimension": 384,
            "max_seq_length": 256,
            "normalize_embeddings": True,
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "dimension": 768,
            "max_seq_length": 384,
            "normalize_embeddings": True,
        },
        "BAAI/bge-small-en-v1.5": {
            "dimension": 384,
            "max_seq_length": 512,
            "normalize_embeddings": True,
        },
        "BAAI/bge-base-en-v1.5": {
            "dimension": 768,
            "max_seq_length": 512,
            "normalize_embeddings": True,
        },
        "BAAI/bge-large-en-v1.5": {
            "dimension": 1024,
            "max_seq_length": 512,
            "normalize_embeddings": True,
        },
    }

    # Return model config if exists, otherwise return default config
    return model_configs.get(
        model_name,
        {
            "dimension": 384,  # Default dimension
            "max_seq_length": 512,
            "normalize_embeddings": True,
        },
    )


# Initialize settings instance
settings = EmbedderSettings()
