from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from uuid import UUID

from rag_ingestor.common.models import Embedding
from rag_ingestor.indexer.models import SearchResult, SearchFilter


class VectorDB(ABC):
    """Interface for vector database implementations."""

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the vector database connection and ensure the collection exists.

        Returns:
            bool: True if initialization was successful
        """
        pass

    @abstractmethod
    def add_embeddings(self, embeddings: List[Embedding]) -> int:
        """
        Add embeddings to the vector database.

        Args:
            embeddings: List of embeddings to add

        Returns:
            int: Number of embeddings successfully added
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[SearchFilter] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors in the database.

        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            filter: Optional filter to apply to the search

        Returns:
            List of search results, ordered by similarity
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[Union[str, UUID]]) -> int:
        """
        Delete embeddings by their IDs.

        Args:
            ids: List of embedding IDs to delete

        Returns:
            int: Number of embeddings successfully deleted
        """
        pass

    @abstractmethod
    def delete_by_filter(self, filter: SearchFilter) -> int:
        """
        Delete embeddings that match the given filter.

        Args:
            filter: Filter to apply for deletion

        Returns:
            int: Number of embeddings successfully deleted
        """
        pass

    @abstractmethod
    def count(self, filter: Optional[SearchFilter] = None) -> int:
        """
        Count embeddings in the database, optionally filtered.

        Args:
            filter: Optional filter to count only matching embeddings

        Returns:
            int: Number of embeddings
        """
        pass


class IndexerServiceInterface(ABC):
    """Interface for indexer service."""

    @abstractmethod
    def index_embeddings(self, embeddings: List[Embedding], **kwargs) -> Dict[str, Any]:
        """
        Index a batch of embeddings.

        Args:
            embeddings: List of embeddings to index
            **kwargs: Additional arguments

        Returns:
            Dict with indexing results
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Union[Dict[str, Any], SearchFilter]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            filter: Optional filter to apply to the search
            **kwargs: Additional arguments

        Returns:
            List of search results, ordered by similarity
        """
        pass

    @abstractmethod
    def delete_embeddings(self, ids: List[Union[str, UUID]]) -> Dict[str, Any]:
        """
        Delete embeddings by their IDs.

        Args:
            ids: List of embedding IDs to delete

        Returns:
            Dict with deletion results
        """
        pass

    @abstractmethod
    async def batch_index_embeddings(
        self, embeddings: List[Embedding], batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Index a large batch of embeddings asynchronously.

        Args:
            embeddings: List of embeddings to index
            batch_size: Size of batches to process

        Returns:
            Dict with indexing results
        """
        pass

    @abstractmethod
    async def index_single_embedding(self, embedding: Embedding) -> bool:
        """
        Index a single embedding asynchronously.

        Args:
            embedding: Embedding to index

        Returns:
            bool: True if indexing was successful
        """
        pass
