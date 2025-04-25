from abc import ABC, abstractmethod
from typing import List, Dict, Any
from rag_ingestor.common.models import Document
from rag_ingestor.embedder.models import EmbeddingRequest, EmbeddingResponse


class TextChunker(ABC):
    """Interface for text chunking strategies."""

    @abstractmethod
    def chunk_text(
        self, text: str, metadata: Dict[str, Any], chunk_size: int, chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with specified size and overlap.

        Args:
            text: The text to be chunked
            metadata: Metadata associated with the text
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        pass


class EmbeddingModel(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector as a list of floats
        """
        pass

    @abstractmethod
    def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts in batch.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors
        """
        pass


class EmbedderServiceInterface(ABC):
    """Interface for embedder service."""

    @abstractmethod
    def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Create embeddings for the document in the request.

        Args:
            request: Embedding request containing document and chunking params

        Returns:
            Response with generated embeddings
        """
        pass

    @abstractmethod
    async def create_embeddings_stream(
        self, document: Document, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> EmbeddingResponse:
        """
        Create embeddings for the document in streaming mode.

        Args:
            document: Document to generate embeddings for
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            Response with generated embeddings
        """
        pass
