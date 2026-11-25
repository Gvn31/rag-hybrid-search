
import json
import math
from pathlib import Path

import chromadb

from src.logger import logger
from src.exception import RAGPipelineException


# ChromaDB configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
COLLECTION_NAME = "rag_documents"


# Initialize persistent ChromaDB storage
def initialize_vector_store():
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

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
        raise RAGPipelineException(str(e)) from e


# Convert metadata into ChromaDB-compatible scalar values
def prepare_metadata(metadata):
    try:
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        prepared_metadata = {}

        for key, value in metadata.items():

            if not isinstance(key, str) or not key.strip():
                raise ValueError("Metadata keys must be non-empty strings")

            if value is None:
                continue

            if isinstance(value, bool):
                prepared_metadata[key] = value

            elif isinstance(value, int):
                prepared_metadata[key] = value

            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Metadata field '{key}' contains an invalid number"
                    )

                prepared_metadata[key] = value

            elif isinstance(value, str):
                prepared_metadata[key] = value

            else:
                # Convert nested values into JSON strings
                prepared_metadata[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str
                )

        if not prepared_metadata:
            prepared_metadata["source"] = "unknown"

        return prepared_metadata

    except Exception as e:
        logger.exception("Metadata preparation failed")
        raise RAGPipelineException(str(e)) from e


# Validate and normalize an embedding vector
def validate_embedding(embedding, field_name="Embedding"):
    try:
        if not isinstance(embedding, (list, tuple)) or not embedding:
            raise ValueError(
                f"{field_name} must be a non-empty list or tuple"
            )

        vector = [float(value) for value in embedding]

        if not all(math.isfinite(value) for value in vector):
            raise ValueError(
                f"{field_name} contains invalid numeric values"
            )

        return vector

    except Exception as e:
        raise RAGPipelineException(str(e)) from e


# Build metadata filters for ChromaDB
def build_metadata_filter(where=None, user_id=None):
    try:
        filters = []

        if where is not None:
            if not isinstance(where, dict) or not where:
                raise ValueError(
                    "where must be a non-empty dictionary"
                )

            filters.append(where)

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string")

            filters.append({"user_id": user_id})

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    except Exception as e:
        logger.exception("Failed to build metadata filter")
        raise RAGPipelineException(str(e)) from e


# Store generated embeddings, text and metadata in ChromaDB
def store_embeddings(embedded_chunks, user_id=None):
    try:
        if not isinstance(embedded_chunks, list):
            raise ValueError(
                "Embedded chunks must be provided as a list"
            )

        if not embedded_chunks:
            logger.warning("No embeddings received for storage")
            return 0

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string")

        _, collection = initialize_vector_store()

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        seen_ids = set()
        expected_dimension = None

        # Validate all chunks before writing to ChromaDB
        for chunk in embedded_chunks:

            if not isinstance(chunk, dict):
                raise ValueError(
                    "Each embedded chunk must be a dictionary"
                )

            required_keys = ["text", "metadata", "embedding"]

            if not all(key in chunk for key in required_keys):
                raise ValueError(
                    "Each chunk must contain text, metadata and embedding"
                )

            text = chunk["text"]
            metadata = chunk["metadata"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chunk text cannot be empty")

            if not isinstance(metadata, dict):
                raise ValueError("Chunk metadata must be a dictionary")

            chunk_metadata = prepare_metadata(metadata)

            chunk_id = chunk_metadata.get("chunk_id")
            source = chunk_metadata.get("source")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    "Chunk metadata must contain a valid chunk_id"
                )

            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    "Chunk metadata must contain a valid source"
                )

            # Add tenant information when supplied
            if user_id is not None:
                chunk_metadata["user_id"] = user_id

            # Create a tenant-specific storage ID
            storage_id = (
                f"{user_id}_{chunk_id}"
                if user_id is not None
                else chunk_id
            )

            if storage_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID detected: {storage_id}"
                )

            seen_ids.add(storage_id)

            vector = validate_embedding(
                chunk["embedding"],
                field_name=f"Embedding for chunk {chunk_id}"
            )

            if expected_dimension is None:
                expected_dimension = len(vector)

            elif len(vector) != expected_dimension:
                raise ValueError(
                    "All embeddings in a batch must have the same dimension"
                )

            ids.append(storage_id)
            documents.append(text)
            metadatas.append(chunk_metadata)
            embeddings.append(vector)

        # Insert new chunks or update existing chunks
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
        raise RAGPipelineException(str(e)) from e


# Retrieve similar document chunks using vector similarity
def search_similar_chunks(
    query_embedding,
    top_k=5,
    user_id=None,
    where=None
):
    try:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_vector = validate_embedding(
            query_embedding,
            field_name="Query embedding"
        )

        # Build tenant-aware metadata filters
        where_filter = build_metadata_filter(
            where=where,
            user_id=user_id
        )

        _, collection = initialize_vector_store()

        total_documents = collection.count()

        if total_documents == 0:
            logger.warning("ChromaDB collection is empty")
            return []

        query_arguments = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, total_documents),
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        if where_filter is not None:
            query_arguments["where"] = where_filter

        # Perform cosine similarity search
        results = collection.query(**query_arguments)

        retrieved_chunks = []

        result_ids = results.get("ids") or []

        if result_ids and result_ids[0]:

            documents = results.get("documents") or [[]]
            metadatas = results.get("metadatas") or [[]]
            distances = results.get("distances") or [[]]

            for index, chunk_id in enumerate(result_ids[0]):

                retrieved_chunks.append({
                    "id": chunk_id,
                    "text": documents[0][index],
                    "metadata": metadatas[0][index],
                    "distance": distances[0][index]
                })

        logger.info(
            f"Retrieved {len(retrieved_chunks)} similar chunks"
        )

        return retrieved_chunks

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Vector similarity search failed")
        raise RAGPipelineException(str(e)) from e


# Delete chunks belonging to a particular document
def delete_document(source, user_id=None):
    try:
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "A valid document source is required"
            )

        # Build source and tenant filters
        where_filter = build_metadata_filter(
            where={"source": source},
            user_id=user_id
        )

        _, collection = initialize_vector_store()

        existing_chunks = collection.get(
            where=where_filter,
            include=["metadatas"]
        )

        deleted_count = len(existing_chunks.get("ids", []))

        if deleted_count == 0:
            logger.warning(
                f"No chunks found for document: {source}"
            )
            return 0

        collection.delete(where=where_filter)

        logger.info(
            f"Deleted {deleted_count} chunks for document: {source}"
        )

        return deleted_count

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Document deletion failed")
        raise RAGPipelineException(str(e)) from e


# Delete selected stored chunk IDs
def delete_chunks_by_ids(chunk_ids):
    try:
        if not isinstance(chunk_ids, (list, set, tuple)):
            raise ValueError("chunk_ids must be a list, set or tuple")

        ids = list(dict.fromkeys(chunk_ids))
        if not all(isinstance(item, str) and item.strip() for item in ids):
            raise ValueError("chunk_ids must contain non-empty strings")

        if not ids:
            return 0

        _, collection = initialize_vector_store()
        existing = collection.get(ids=ids)
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

        return len(existing_ids)

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Selected ChromaDB chunk deletion failed")
        raise RAGPipelineException(str(e)) from e


# Return the total number of stored chunks
def get_collection_count():
    try:
        _, collection = initialize_vector_store()

        total_chunks = collection.count()

        logger.info(
            f"Total chunks in ChromaDB: {total_chunks}"
        )

        return total_chunks

    except Exception as e:
        logger.exception("Failed to get collection count")
        raise RAGPipelineException(str(e)) from e


# Test ChromaDB storage and retrieval independently
def main():
    try:
        from src.embeddings import generate_embeddings, get_embedding_model

        sample_texts = [
            (
                "Artificial intelligence enables machines "
                "to perform tasks requiring human intelligence."
            ),
            (
                "Machine learning allows computers to learn "
                "patterns from data."
            )
        ]

        sample_chunks = []

        current_position = 0

        for index, text in enumerate(sample_texts):

            start_index = current_position
            end_index = start_index + len(text)

            sample_chunks.append({
                "text": text,
                "metadata": {
                    "source": "vector_store_test.txt",
                    "chunk_id": f"vector_test_{index + 1:03d}",
                    "chunk_index": index,
                    "chunking_strategy": "test",
                    "char_count": len(text),
                    "start_index": start_index,
                    "end_index": end_index,
                    "section_heading": None
                }
            })

            current_position = end_index + 1

        # Generate document embeddings
        embedded_chunks = generate_embeddings(sample_chunks)

        # Store document embeddings
        stored_count = store_embeddings(embedded_chunks)

        print(f"\nStored chunks: {stored_count}")

        # Generate an embedding for the search query
        query_text = "How do computers learn from data?"

        embedding_model = get_embedding_model()

        query_embedding = embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].tolist()

        # Retrieve similar chunks
        search_results = search_similar_chunks(
            query_embedding=query_embedding,
            top_k=2
        )

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

        # Remove only the test document
        deleted_count = delete_document(
            source="vector_store_test.txt"
        )

        print(f"\nTest chunks deleted: {deleted_count}")

    except RAGPipelineException as e:
        logger.error(f"Vector store test failed: {e}")
        print(f"Vector store test failed: {e}")

    except Exception as e:
        logger.exception(
            "Unexpected error in vector store module"
        )
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
