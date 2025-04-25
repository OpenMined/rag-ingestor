import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
import numpy as np

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_ingestor.common.models import Document, Embedding
from rag_ingestor.common.config import settings
from rag_ingestor.embedder.models import EmbeddingRequest, EmbeddingResponse
from rag_ingestor.embedder.interfaces import (
    TextChunker,
    EmbeddingModel,
    EmbedderServiceInterface,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaIndexChunker(TextChunker):
    """Text chunker implementation using llama-index SimpleNodeParser."""

    def __init__(self):
        self.node_parser = SimpleNodeParser.from_defaults()

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks using llama-index SimpleNodeParser.

        Args:
            text: The text to be chunked
            metadata: Metadata associated with the text
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Configure parser with specified parameters
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Create llama-index document
        llama_doc = LlamaDocument(
            text=text,
            metadata=metadata,
        )

        # Parse document into nodes
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])

        # Convert nodes to required format
        chunks = []
        for node in nodes:
            chunk_data = {"text": node.text, "metadata": {**metadata, **node.metadata}}
            chunks.append(chunk_data)

        return chunks

    def stream_chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        buffer_size: int = 10000,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream text chunking for large documents.

        Args:
            text: The text to be chunked
            metadata: Metadata associated with the text
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            buffer_size: Size of text buffer to process at once

        Yields:
            Dictionaries containing chunk text and metadata
        """
        # For very large texts, process in buffer-sized segments
        text_length = len(text)

        for start_idx in range(0, text_length, buffer_size - chunk_size):
            end_idx = min(start_idx + buffer_size, text_length)
            buffer_text = text[start_idx:end_idx]

            # Process this buffer
            buffer_chunks = self.chunk_text(
                buffer_text,
                {**metadata, "buffer_start_idx": start_idx, "buffer_end_idx": end_idx},
                chunk_size,
                chunk_overlap,
            )

            # Yield each chunk
            for chunk in buffer_chunks:
                yield chunk


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Embedding model implementation using HuggingFace models."""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the HuggingFace embedding model
        """
        self.model_name = model_name or settings.embedding_model
        self.model = HuggingFaceEmbedding(model_name=self.model_name)
        logger.info(
            f"Initialized HuggingFaceEmbeddingModel with model: {self.model_name}"
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Text to generate embedding for

        Returns:
            Embedding vector as a list of floats
        """
        try:
            return self.model.get_text_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts in batch.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors
        """
        try:
            # Check if the model supports batch processing
            if hasattr(self.model, "get_text_embeddings"):
                return self.model.get_text_embeddings(texts)

            # Fall back to sequential processing if batch not supported
            return [self.get_embedding(text) for text in texts]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    @property
    def dimension(self) -> int:
        """
        Return the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors
        """
        # Generate an embedding for a short text to determine dimension
        sample_embedding = self.get_embedding("dimension test")
        return len(sample_embedding)


class EmbedderService(EmbedderServiceInterface):
    """Service for creating document embeddings."""

    def __init__(
        self, chunker: TextChunker = None, embedding_model: EmbeddingModel = None
    ):
        """
        Initialize the embedder service.

        Args:
            chunker: Text chunker implementation
            embedding_model: Embedding model implementation
        """
        self.chunker = chunker or LlamaIndexChunker()
        self.embedding_model = embedding_model or HuggingFaceEmbeddingModel()
        logger.info("EmbedderService initialized")

    def create_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Create embeddings for the document in the request.

        Args:
            request: Embedding request containing document and chunking params

        Returns:
            Response with generated embeddings
        """
        try:
            # Validate document
            if not request.document or not request.document.content:
                raise ValueError("Document or document content is missing")

            # Extract parameters
            document = request.document
            chunk_size = request.chunk_size
            chunk_overlap = request.chunk_overlap

            # Get text chunks
            chunks = self.chunker.chunk_text(
                document.content, document.metadata or {}, chunk_size, chunk_overlap
            )

            # Prepare texts for batch embedding
            texts = [chunk["text"] for chunk in chunks]

            # Generate embeddings in batch
            embedding_vectors = self.embedding_model.batch_get_embeddings(texts)

            # Create embedding objects
            embeddings = []
            for i, (chunk, vector) in enumerate(zip(chunks, embedding_vectors)):
                embedding = Embedding(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    document_id=document.id,
                    metadata={"chunk_text": chunk["text"], **chunk["metadata"]},
                )
                embeddings.append(embedding)

            return EmbeddingResponse(
                embeddings=embeddings, total_chunks=len(embeddings)
            )

        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    async def create_embeddings_stream(
        self, document: Document, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> AsyncIterator[Embedding]:
        """
        Create embeddings for the document in streaming mode.

        Args:
            document: Document to generate embeddings for
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between consecutive chunks

        Yields:
            Generated embeddings one by one
        """
        try:
            # Validate document
            if not document or not document.content:
                raise ValueError("Document or document content is missing")

            # Use synchronous generator for chunking
            chunk_generator = self.chunker.stream_chunk_text(
                document.content, document.metadata or {}, chunk_size, chunk_overlap
            )

            # Process chunks in batches for efficiency
            batch_size = 10
            batch = []

            for chunk in chunk_generator:
                batch.append(chunk)

                # Process when batch is full
                if len(batch) >= batch_size:
                    # Get texts from batch
                    texts = [c["text"] for c in batch]

                    # Generate embeddings in batch
                    embedding_vectors = self.embedding_model.batch_get_embeddings(texts)

                    # Create and yield embedding objects
                    for chunk_data, vector in zip(batch, embedding_vectors):
                        embedding = Embedding(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            document_id=document.id,
                            metadata={
                                "chunk_text": chunk_data["text"],
                                **chunk_data["metadata"],
                            },
                        )
                        yield embedding

                        # Small pause to allow other tasks to run
                        await asyncio.sleep(0.001)

                    # Clear batch
                    batch = []

            # Process remaining chunks
            if batch:
                texts = [c["text"] for c in batch]
                embedding_vectors = self.embedding_model.batch_get_embeddings(texts)

                for chunk_data, vector in zip(batch, embedding_vectors):
                    embedding = Embedding(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        document_id=document.id,
                        metadata={
                            "chunk_text": chunk_data["text"],
                            **chunk_data["metadata"],
                        },
                    )
                    yield embedding
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in streaming embeddings: {str(e)}")
            raise


if __name__ == "__main__":
    # Test the service with dummy data
    from uuid import uuid4

    service = EmbedderService()
    document = Document(id=uuid4(), content="Hello, world!")
    response = service.create_embeddings(EmbeddingRequest(document=document))
    print(response)

    # Test streaming embeddings
    async def test_streaming():
        stream_response = service.create_embeddings_stream(document)
        async for embedding in stream_response:
            print(embedding.document_id)
            print(embedding.metadata)
            print(embedding.vector)

    asyncio.run(test_streaming())
