import logging
from typing import List, Dict, Any, Optional, Union
import time
from uuid import UUID

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.exceptions import UnexpectedResponse

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from rag_ingestor.common.models import Embedding
from rag_ingestor.indexer.config import settings
from rag_ingestor.indexer.models import SearchResult, SearchFilter
from rag_ingestor.indexer.interfaces import VectorDB


logger = logging.getLogger(__name__)


class QdrantDB(VectorDB):
    """Implementation of VectorDB interface using Qdrant."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Qdrant database adapter.

        Args:
            config: Optional configuration dictionary that can override settings from config module
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client is not installed. Please install it with: pip install qdrant-client"
            )

        # Store the config
        self.config = config or {}

        # Get settings from config or fall back to indexer settings
        self.host = self.config.get("vector_db_host", settings.vector_db_host)
        self.port = self.config.get("vector_db_port", settings.vector_db_port)
        self.api_key = self.config.get("vector_db_api_key", settings.vector_db_api_key)
        self.collection_name = self.config.get(
            "vector_db_collection", settings.vector_db_collection
        )
        self.dimension = self.config.get(
            "embedding_dimension", settings.embedding_dimension
        )

        # Additional settings with defaults
        similarity_metric = self.config.get(
            "similarity_metric", settings.similarity_metric
        )
        self.similarity = self._map_similarity_metric(similarity_metric)
        self.prefer_grpc = self.config.get(
            "qdrant_prefer_grpc", settings.qdrant_prefer_grpc
        )
        self.timeout = self.config.get("qdrant_timeout", settings.qdrant_timeout)

        # Initialize client
        self.client = None
        self._create_client()

    def _create_client(self) -> None:
        """Create Qdrant client based on configuration."""
        try:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout,
            )
            logger.info(f"Created Qdrant client: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Error creating Qdrant client: {str(e)}")
            raise

    def _map_similarity_metric(self, metric: str) -> qdrant_models.Distance:
        """Map similarity metric string to Qdrant distance enum."""
        mapping = {
            "Cosine": qdrant_models.Distance.COSINE,
            "Dot": qdrant_models.Distance.DOT,
            "Euclidean": qdrant_models.Distance.EUCLID,
        }
        if metric not in mapping:
            logger.warning(
                f"Unknown similarity metric: {metric}, using Cosine as default"
            )
            return qdrant_models.Distance.COSINE
        return mapping[metric]

    def _convert_filter(
        self, filter: Optional[SearchFilter]
    ) -> Optional[qdrant_models.Filter]:
        """Convert SearchFilter to Qdrant filter format."""
        if filter is None:
            return None

        conditions = []

        # Process exact match filters
        for field, value in filter.exact_filters.items():
            if isinstance(value, list):
                # Handle list of values (OR condition)
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=field, match=qdrant_models.MatchAny(any=value)
                    )
                )
            else:
                # Handle single value (support UUID by converting to string)
                if isinstance(value, UUID):
                    value = str(value)
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=field, match=qdrant_models.MatchValue(value=value)
                    )
                )

        # Process range filters
        for field, range_values in filter.range_filters.items():
            range_condition = {}

            if "gt" in range_values:
                range_condition["gt"] = range_values["gt"]
            if "gte" in range_values:
                range_condition["gte"] = range_values["gte"]
            if "lt" in range_values:
                range_condition["lt"] = range_values["lt"]
            if "lte" in range_values:
                range_condition["lte"] = range_values["lte"]

            if range_condition:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key=field, range=qdrant_models.Range(**range_condition)
                    )
                )

        if not conditions:
            return None

        return qdrant_models.Filter(must=conditions)

    def initialize(self) -> bool:
        """
        Initialize Qdrant collection.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                # Create collection if it doesn't exist
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.dimension, distance=self.similarity
                    ),
                )
                logger.info(
                    f"Created collection '{self.collection_name}' with dimension {self.dimension}"
                )

            self.client.update_collection(
                collection_name=self.collection_name,
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=1,
                ),
            )

            return True
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {str(e)}")
            return False

    def add_embeddings(self, embeddings: List[Embedding]) -> int:
        """
        Add embeddings to Qdrant.

        Args:
            embeddings: List of embeddings to add

        Returns:
            int: Number of embeddings successfully added
        """
        if not embeddings:
            return 0

        try:
            # Prepare points for batch upload
            ids = []
            vectors = []
            payloads = []

            for emb in embeddings:
                # Convert UUID to string if needed
                emb_id = str(emb.id) if isinstance(emb.id, UUID) else emb.id
                doc_id = (
                    str(emb.document_id)
                    if isinstance(emb.document_id, UUID)
                    else emb.document_id
                )

                # Validate vector dimension
                if len(emb.vector) != self.dimension:
                    raise ValueError(
                        f"Vector dimension must be {self.dimension}, got {len(emb.vector)} for embedding {emb_id}"
                    )

                ids.append(emb_id)
                vectors.append(emb.vector)

                # Prepare payload with document_id and metadata
                payload = {"document_id": doc_id, **emb.metadata}
                payloads.append(payload)

            # Upload batch to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_models.Batch(ids=ids, vectors=vectors, payloads=payloads),
            )

            return len(embeddings)
        except Exception as e:
            logger.error(f"Error adding embeddings to Qdrant: {str(e)}")
            raise

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[SearchFilter] = None,
    ) -> List[SearchResult]:
        """
        Search for similar vectors in Qdrant.

        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            filter: Optional filter to apply to the search

        Returns:
            List of search results, ordered by similarity
        """
        try:
            # Validate query vector dimension
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Query vector dimension must be {self.dimension}, got {len(query_vector)}"
                )

            # Convert filter to Qdrant format
            qdrant_filter = self._convert_filter(filter)

            # Execute search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

            # Convert to SearchResult objects
            search_results = []
            for res in results:
                document_id = res.payload.get("document_id", "")
                metadata = {k: v for k, v in res.payload.items() if k != "document_id"}

                search_result = SearchResult(
                    id=res.id,
                    document_id=document_id,
                    score=res.score,
                    metadata=metadata,
                    content=metadata.get("chunk_text"),
                )
                search_results.append(search_result)

            return search_results
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            raise

    def delete_by_ids(self, ids: List[Union[str, UUID]]) -> int:
        """
        Delete embeddings by their IDs.

        Args:
            ids: List of embedding IDs to delete

        Returns:
            int: Number of embeddings successfully deleted
        """
        if not ids:
            return 0

        try:
            # Convert UUID to string if needed
            str_ids = [str(id) if isinstance(id, UUID) else id for id in ids]

            # Delete points by IDs
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(points=str_ids),
            )

            # In Qdrant, a successful deletion returns operation_id, not count
            # So we just return the length of the input list
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting embeddings from Qdrant: {str(e)}")
            raise

    def delete_by_filter(self, filter: SearchFilter) -> int:
        """
        Delete embeddings that match the given filter.

        Args:
            filter: Filter to apply for deletion

        Returns:
            int: Number of embeddings successfully deleted
        """
        try:
            # First count how many will be deleted
            count_before = self.count()

            # Convert filter to Qdrant format
            qdrant_filter = self._convert_filter(filter)

            if qdrant_filter is None:
                logger.warning("No filter conditions provided for delete_by_filter")
                return 0

            # Delete points by filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.FilterSelector(filter=qdrant_filter),
            )

            # Calculate how many were deleted by comparing counts before and after
            count_after = self.count()
            deleted_count = count_before - count_after

            return max(0, deleted_count)  # Ensure we don't return negative numbers
        except Exception as e:
            logger.error(f"Error deleting embeddings by filter from Qdrant: {str(e)}")
            raise

    def count(self, filter: Optional[SearchFilter] = None) -> int:
        """
        Count embeddings in Qdrant, optionally filtered.

        Args:
            filter: Optional filter to count only matching embeddings

        Returns:
            int: Number of embeddings
        """
        try:
            # Convert filter to Qdrant format
            qdrant_filter = self._convert_filter(filter)

            # Get collection info
            if qdrant_filter is None:
                # If no filter, get the total count from collection info
                collection_info = self.client.get_collection(self.collection_name)
                vector_count = collection_info.points_count
                return vector_count if vector_count is not None else 0
            else:
                # If filter provided, count with filter
                count_result = self.client.count(
                    collection_name=self.collection_name, count_filter=qdrant_filter
                )
                return count_result.count
        except Exception as e:
            logger.error(f"Error counting embeddings in Qdrant: {str(e)}")
            raise
