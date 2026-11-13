
import sys
from pathlib import Path

import chromadb

from src.logger import logger
from src.exception import RAGPipelineException


# Define the persistent ChromaDB storage directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"

# Define the ChromaDB collection name
COLLECTION_NAME = "rag_documents"


# Initialize ChromaDB and create or access the collection
def initialize_vector_store():
    try:
        # Create the storage directory if it does not exist
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize persistent ChromaDB client
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        # Create or access the document collection
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "RAG document embeddings",
                "hnsw:space": "cosine"
            }
        )

        logger.info("ChromaDB initialized successfully")

        return client, collection

    except Exception as e:
        logger.exception("Failed to initialize ChromaDB")
        raise RAGPipelineException(str(e))


# Convert metadata values into ChromaDB-compatible scalar values
def prepare_metadata(metadata):
    try:
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        prepared_metadata = {}

        for key, value in metadata.items():

            # ChromaDB does not accept None or nested metadata values
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                prepared_metadata[key] = value

            else:
                prepared_metadata[key] = str(value)

        # ChromaDB requires metadata to contain at least one field
        if not prepared_metadata:
            prepared_metadata["source"] = "unknown"

        return prepared_metadata

    except Exception as e:
        logger.exception("Metadata preparation failed")
        raise RAGPipelineException(str(e))


# Store generated embeddings, text and metadata in ChromaDB
def store_embeddings(embedded_chunks, user_id=None):
    try:
        # Validate input
        if not isinstance(embedded_chunks, list):
            raise ValueError("Embedded chunks must be provided as a list")

        if not embedded_chunks:
            logger.warning("No embeddings received for storage")
            return 0

        # Initialize ChromaDB collection
        _, collection = initialize_vector_store()

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        # Prepare each embedded chunk for storage
        for chunk in embedded_chunks:

            if not isinstance(chunk, dict):
                raise ValueError("Each embedded chunk must be a dictionary")

            if not all(
                key in chunk for key in ["text", "metadata", "embedding"]
            ):
                raise ValueError(
                    "Each chunk must contain text, metadata and embedding"
                )

            text = chunk["text"]
            metadata = chunk["metadata"]
            embedding = chunk["embedding"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chunk text cannot be empty")

            if not isinstance(metadata, dict):
                raise ValueError("Chunk metadata must be a dictionary")

            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Embedding must be a non-empty list")

            # Ensure all vector values are numeric
            vector = [float(value) for value in embedding]

            if not all(
                value == value and abs(value) != float("inf")
                for value in vector
            ):
                raise ValueError("Embedding contains invalid numeric values")

            # Prepare metadata for ChromaDB
            chunk_metadata = prepare_metadata(metadata)

            # Add user ID for document-level tenant isolation
            if user_id is not None:
                chunk_metadata["user_id"] = str(user_id)

            # Generate a stable ID for each stored chunk
            chunk_id = chunk_metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError("Chunk metadata must contain chunk_id")

            # Include user ID in the storage ID to avoid cross-user collisions
            storage_id = (
                f"{user_id}_{chunk_id}"
                if user_id is not None
                else str(chunk_id)
            )

            ids.append(storage_id)
            documents.append(text)
            metadatas.append(chunk_metadata)
            embeddings.append(vector)

        # Store or update chunks in ChromaDB
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        logger.info(
            f"Successfully stored {len(ids)} embeddings in ChromaDB"
        )

        return len(ids)

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to store embeddings")
        raise RAGPipelineException(str(e))


# Retrieve similar document chunks using vector similarity
def search_similar_chunks(
    query_embedding,
    top_k=5,
    user_id=None,
    where=None
):
    try:
        # Validate query embedding
        if not isinstance(query_embedding, list) or not query_embedding:
            raise ValueError("Query embedding must be a non-empty list")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = [float(value) for value in query_embedding]

        if not all(
            value == value and abs(value) != float("inf")
            for value in query_vector
        ):
            raise ValueError("Query embedding contains invalid values")

        # Initialize ChromaDB collection
        _, collection = initialize_vector_store()

        # Return an empty list when the collection has no documents
        total_documents = collection.count()

        if total_documents == 0:
            logger.warning("ChromaDB collection is empty")
            return []

        # Build metadata filters for user-level isolation
        filters = []

        if where is not None:
            if not isinstance(where, dict):
                raise ValueError("where must be a dictionary")

            filters.append(where)

        if user_id is not None:
            filters.append({"user_id": str(user_id)})

        # Combine filters when multiple conditions are provided
        where_filter = None

        if len(filters) == 1:
            where_filter = filters[0]

        elif len(filters) > 1:
            where_filter = {"$and": filters}

        # Search for the nearest vectors
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, total_documents),
            where=where_filter,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        # Convert ChromaDB response into a simple list of chunks
        retrieved_chunks = []

        if results["ids"]:
            for index, chunk_id in enumerate(results["ids"][0]):

                retrieved_chunks.append({
                    "id": chunk_id,
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index]
                })

        logger.info(
            f"Retrieved {len(retrieved_chunks)} similar chunks"
        )

        return retrieved_chunks

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Vector similarity search failed")
        raise RAGPipelineException(str(e))


# Delete stored chunks belonging to a particular document
def delete_document(source, user_id=None):
    try:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A valid document source is required")

        # Initialize ChromaDB collection
        _, collection = initialize_vector_store()

        # Build document deletion filter
        filters = [{"source": source}]

        if user_id is not None:
            filters.append({"user_id": str(user_id)})

        if len(filters) == 1:
            where_filter = filters[0]
        else:
            where_filter = {"$and": filters}

        # Find matching chunks before deleting them
        existing_chunks = collection.get(
            where=where_filter,
            include=["metadatas"]
        )

        deleted_count = len(existing_chunks["ids"])

        if deleted_count == 0:
            logger.warning(f"No chunks found for document: {source}")
            return 0

        # Delete all matching chunks
        collection.delete(
            where=where_filter
        )

        logger.info(
            f"Deleted {deleted_count} chunks for document: {source}"
        )

        return deleted_count

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Document deletion failed")
        raise RAGPipelineException(str(e))


# Return the total number of stored chunks
def get_collection_count():
    try:
        # Initialize ChromaDB collection
        _, collection = initialize_vector_store()

        total_chunks = collection.count()

        logger.info(f"Total chunks in ChromaDB: {total_chunks}")

        return total_chunks

    except Exception as e:
        logger.exception("Failed to get collection count")
        raise RAGPipelineException(str(e))


# Test ChromaDB storage and retrieval independently
def main():
    try:
        from src.embeddings import generate_embeddings

        # Create sample chunks for testing
        sample_chunks = [
            {
                "text": (
                    "Artificial intelligence enables machines "
                    "to perform tasks requiring human intelligence."
                ),
                "metadata": {
                    "source": "vector_store_test.txt",
                    "chunk_id": "vector_test_001",
                    "chunk_index": 0
                }
            },
            {
                "text": (
                    "Machine learning allows computers to learn "
                    "patterns from data."
                ),
                "metadata": {
                    "source": "vector_store_test.txt",
                    "chunk_id": "vector_test_002",
                    "chunk_index": 1
                }
            }
        ]

        # Generate embeddings for the sample chunks
        embedded_chunks = generate_embeddings(sample_chunks)

        # Store the generated embeddings in ChromaDB
        stored_count = store_embeddings(embedded_chunks)

        print(f"\nStored chunks: {stored_count}")

        # Generate an embedding for the search query
        query_text = "How do computers learn from data?"

        query_chunks = [
            {
                "text": query_text,
                "metadata": {
                    "source": "query"
                }
            }
        ]

        query_embedding = generate_embeddings(query_chunks)[0]["embedding"]

        # Search for similar document chunks
        search_results = search_similar_chunks(
            query_embedding=query_embedding,
            top_k=2
        )

        # Display retrieved results
        print("\nSearch Results:")

        for result in search_results:
            print("\nChunk ID:", result["id"])
            print("Text:", result["text"])
            print("Distance:", result["distance"])
            print("Metadata:", result["metadata"])

        # Display total stored chunks
        print(
            "\nTotal chunks in collection:",
            get_collection_count()
        )

        # Clean up only the test document
        deleted_count = delete_document(
            source="vector_store_test.txt"
        )

        print(f"\nTest chunks deleted: {deleted_count}")

    except RAGPipelineException as e:
        logger.error(f"Vector store test failed: {e}")
        print(f"Vector store test failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in vector store module")
        print(f"Unexpected error: {e}")


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()


