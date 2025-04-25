import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import time
from uuid import UUID

from rag_ingestor.common.models import Embedding
from rag_ingestor.indexer.config import settings
from rag_ingestor.indexer.models import SearchResult, SearchFilter
from rag_ingestor.indexer.interfaces import VectorDB, IndexerServiceInterface
from rag_ingestor.indexer.qdrant import QdrantDB


logger = logging.getLogger(__name__)


class IndexerService(IndexerServiceInterface):
    """Service for indexing and searching embeddings."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vector_db: Optional[VectorDB] = None,
    ):
        """
        Initialize the indexer service.

        Args:
            config: Optional configuration dictionary that can override settings from config module
            vector_db: Optional pre-configured vector database client
        """
        self.config = config or {}
        self.vector_db = vector_db

        if vector_db is None:
            # Initialize vector database client with Qdrant
            self.vector_db = QdrantDB(self.config)

        # Initialize the database
        initialized = self.vector_db.initialize()
        if not initialized:
            logger.error("Failed to initialize vector database")
            raise RuntimeError("Failed to initialize vector database")

        logger.info("IndexerService initialized")

    def index_embeddings(self, embeddings: List[Embedding], **kwargs) -> Dict[str, Any]:
        """
        Index a batch of embeddings.

        Args:
            embeddings: List of embeddings to index
            **kwargs: Additional arguments (unused)

        Returns:
            Dict with indexing results
        """
        start_time = time.time()

        try:
            if not embeddings:
                return {"success": True, "indexed_count": 0, "duration_ms": 0}

            # Use the vector_db to add embeddings
            indexed_count = self.vector_db.add_embeddings(embeddings)

            duration_ms = int((time.time() - start_time) * 1000)

            result = {
                "success": True,
                "indexed_count": indexed_count,
                "duration_ms": duration_ms,
            }

            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(f"Error indexing embeddings: {str(e)}")

            result = {
                "success": False,
                "error": str(e),
                "indexed_count": 0,
                "duration_ms": duration_ms,
            }

            return result

    def search(
        self,
        query_vector: List[float],
        top_k: int = None,
        filter: Optional[Union[Dict[str, Any], SearchFilter]] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Vector to search for
            top_k: Number of results to return (defaults to config value)
            filter: Optional filter to apply to the search (Dict or SearchFilter)
            **kwargs: Additional arguments (unused)

        Returns:
            List of search results, ordered by similarity
        """
        try:
            start_time = time.time()

            # Use configured default if top_k is not provided
            if top_k is None:
                top_k = self.config.get("default_top_k", settings.default_top_k)

            # Convert dict filter to SearchFilter if needed
            search_filter = None
            if filter is not None:
                if isinstance(filter, dict):
                    search_filter = SearchFilter(exact_filters=filter)
                else:
                    search_filter = filter

            # Use the vector_db to search
            results = self.vector_db.search(
                query_vector=query_vector, top_k=top_k, filter=search_filter
            )

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Search completed in {duration_ms}ms, found {len(results)} results"
            )

            return results
        except Exception as e:
            logger.error(f"Error searching for embeddings: {str(e)}")
            raise

    def delete_embeddings(self, ids: List[Union[str, UUID]]) -> Dict[str, Any]:
        """
        Delete embeddings by their IDs.

        Args:
            ids: List of embedding IDs to delete

        Returns:
            Dict with deletion results
        """
        start_time = time.time()

        try:
            if not ids:
                return {"success": True, "deleted_count": 0, "duration_ms": 0}

            # Delete embeddings
            deleted_count = self.vector_db.delete_by_ids(ids)

            duration_ms = int((time.time() - start_time) * 1000)

            result = {
                "success": True,
                "deleted_count": deleted_count,
                "duration_ms": duration_ms,
            }

            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(f"Error deleting embeddings: {str(e)}")

            result = {
                "success": False,
                "error": str(e),
                "deleted_count": 0,
                "duration_ms": duration_ms,
            }

            return result

    def delete_by_filter(
        self, filter: Union[Dict[str, Any], SearchFilter]
    ) -> Dict[str, Any]:
        """
        Delete embeddings by filter.

        Args:
            filter: Filter to apply for deletion

        Returns:
            Dict with deletion results
        """
        start_time = time.time()

        try:
            # Convert dict filter to SearchFilter if needed
            search_filter = None
            if isinstance(filter, dict):
                search_filter = SearchFilter(exact_filters=filter)
            else:
                search_filter = filter

            # Delete embeddings
            deleted_count = self.vector_db.delete_by_filter(search_filter)

            duration_ms = int((time.time() - start_time) * 1000)

            result = {
                "success": True,
                "deleted_count": deleted_count,
                "duration_ms": duration_ms,
            }

            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(f"Error deleting embeddings by filter: {str(e)}")

            result = {
                "success": False,
                "error": str(e),
                "deleted_count": 0,
                "duration_ms": duration_ms,
            }

            return result

    async def batch_index_embeddings(
        self, embeddings: List[Embedding], batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Index a large batch of embeddings asynchronously.

        Args:
            embeddings: List of embeddings to index
            batch_size: Size of batches to process (defaults to config value)

        Returns:
            Dict with indexing results
        """
        start_time = time.time()

        try:
            if not embeddings:
                return {
                    "success": True,
                    "indexed_count": 0,
                    "batch_count": 0,
                    "duration_ms": 0,
                }

            # Use configured default if batch_size is not provided
            if batch_size is None:
                batch_size = self.config.get("batch_size", settings.batch_size)

            # Split embeddings into batches
            batches = [
                embeddings[i : i + batch_size]
                for i in range(0, len(embeddings), batch_size)
            ]

            # Process batches with small delay between them
            total_indexed = 0
            for batch in batches:
                # Process batch
                indexed_count = self.vector_db.add_embeddings(batch)
                total_indexed += indexed_count

                # Small delay to prevent overwhelming the database
                await asyncio.sleep(0.1)

            duration_ms = int((time.time() - start_time) * 1000)

            result = {
                "success": True,
                "indexed_count": total_indexed,
                "batch_count": len(batches),
                "duration_ms": duration_ms,
            }

            return result
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(f"Error in batch indexing: {str(e)}")

            result = {
                "success": False,
                "error": str(e),
                "indexed_count": 0,
                "batch_count": 0,
                "duration_ms": duration_ms,
            }

            return result

    async def index_single_embedding(self, embedding: Embedding) -> bool:
        """
        Index a single embedding asynchronously.

        Args:
            embedding: Embedding to index

        Returns:
            bool: True if indexing was successful
        """
        try:
            # Add the single embedding
            indexed_count = self.vector_db.add_embeddings([embedding])
            return indexed_count == 1
        except Exception as e:
            logger.error(f"Error indexing single embedding: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.

        Returns:
            Dict with statistics
        """
        try:
            # Get total count
            total_count = self.vector_db.count()

            stats = {"total_embeddings": total_count, "status": "healthy"}

            # Add collection name if available
            if hasattr(self.vector_db, "collection_name"):
                stats["collection_name"] = self.vector_db.collection_name

            return stats
        except Exception as e:
            logger.error(f"Error getting vector DB stats: {str(e)}")
            return {"status": "error", "error": str(e), "total_embeddings": 0}


if __name__ == "__main__":
    import uuid
    import numpy as np
    from rag_ingestor.indexer.config import settings

    indexer_service = IndexerService()
    result = indexer_service.index_embeddings(
        [
            Embedding(
                vector=np.random.rand(settings.embedding_dimension).tolist(),
                document_id=uuid.uuid4(),
            )
        ]
    )
    print(result)

    result = indexer_service.search(
        np.random.rand(settings.embedding_dimension).tolist(), top_k=10
    )
    print(result)
