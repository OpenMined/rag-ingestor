from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Embedder settings
    embedder_host: str = "0.0.0.0"
    embedder_port: int = 8000
    
    # Indexer settings
    indexer_host: str = "0.0.0.0"
    indexer_port: int = 8001
    
    # Main service settings
    main_host: str = "0.0.0.0"
    main_port: int = 8002
    
    # Vector DB settings
    vector_db_host: str = "localhost"
    vector_db_port: int = 6333
    vector_db_collection: str = "documents"
    vector_db_api_key: Optional[str] = None
    
    # Embedding model settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() 