
import numpy as np

from src.logger import logger
from src.exception import RAGPipelineException
from src.vector_store import (
    search_similar_chunks,
)


# Define the cosine similarity threshold
DEDUPLICATION_THRESHOLD = 0.95


# Normalize an embedding vector before similarity comparison
def normalize_embedding(embedding):
    try:
        # Convert the embedding into a NumPy array
        vector = np.asarray(embedding, dtype=np.float32)

        # Validate the embedding dimensions
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Embedding must be a non-empty 1D vector")

        # Validate numeric values
        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding contains invalid numeric values")

        # Calculate the vector magnitude
        magnitude = np.linalg.norm(vector)

        if magnitude == 0:
            raise ValueError("Embedding magnitude cannot be zero")

        # Return the normalized vector
        return vector / magnitude

    except Exception as e:
        logger.exception("Embedding normalization failed")
        raise RAGPipelineException(str(e))


# Calculate cosine similarity between two embedding vectors
def calculate_cosine_similarity(embedding_1, embedding_2):
    try:
        # Normalize both embedding vectors
        vector_1 = normalize_embedding(embedding_1)
        vector_2 = normalize_embedding(embedding_2)

        # Ensure both vectors have the same dimensions
        if vector_1.shape != vector_2.shape:
            raise ValueError(
                "Embedding dimensions do not match"
            )

        # Calculate cosine similarity
        similarity = float(np.dot(vector_1, vector_2))

        # Keep the result within the valid cosine similarity range
        similarity = float(np.clip(similarity, -1.0, 1.0))

        return similarity

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Cosine similarity calculation failed")
        raise RAGPipelineException(str(e))


# Validate incoming embedded chunks
def validate_embedded_chunks(embedded_chunks):
    try:
        if not isinstance(embedded_chunks, list):
            raise ValueError(
                "Embedded chunks must be provided as a list"
            )

        # Validate each embedded chunk
        for chunk in embedded_chunks:

            if not isinstance(chunk, dict):
                raise ValueError(
                    "Each embedded chunk must be a dictionary"
                )

            if not all(
                key in chunk
                for key in ["text", "metadata", "embedding"]
            ):
                raise ValueError(
                    "Each chunk must contain text, metadata and embedding"
                )

            if (
                not isinstance(chunk["text"], str)
                or not chunk["text"].strip()
            ):
                raise ValueError("Chunk text cannot be empty")

            if not isinstance(chunk["metadata"], dict):
                raise ValueError("Chunk metadata must be a dictionary")

            if not chunk["metadata"].get("chunk_id"):
                raise ValueError(
                    "Chunk metadata must contain chunk_id"
                )

            # Validate the embedding vector
            normalize_embedding(chunk["embedding"])

        return True

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Embedded chunk validation failed")
        raise RAGPipelineException(str(e))


# Check whether a chunk already exists in ChromaDB
def check_existing_duplicate(
    embedding,
    threshold=DEDUPLICATION_THRESHOLD,
    user_id=None
):
    try:
        # Search for the nearest existing vector
        existing_results = search_similar_chunks(
            query_embedding=embedding,
            top_k=1,
            user_id=user_id
        )

        # No existing vectors means no duplicate
        if not existing_results:
            return {
                "is_duplicate": False,
                "similarity": None,
                "duplicate_of": None
            }

        # Get the closest existing chunk
        nearest_chunk = existing_results[0]

        # ChromaDB cosine distance = 1 - cosine similarity
        distance = float(nearest_chunk["distance"])

        similarity = float(
            np.clip(1.0 - distance, -1.0, 1.0)
        )

        # Identify duplicate chunks using the configured threshold
        if similarity > threshold:
            return {
                "is_duplicate": True,
                "similarity": similarity,
                "duplicate_of": nearest_chunk["id"]
            }

        return {
            "is_duplicate": False,
            "similarity": similarity,
            "duplicate_of": None
        }

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Existing duplicate check failed")
        raise RAGPipelineException(str(e))


# Check whether a chunk duplicates another chunk in the same batch
def check_batch_duplicate(
    embedding,
    accepted_chunks,
    threshold=DEDUPLICATION_THRESHOLD
):
    try:
        # Compare the incoming chunk with previously accepted chunks
        for existing_chunk in accepted_chunks:

            similarity = calculate_cosine_similarity(
                embedding,
                existing_chunk["embedding"]
            )

            # Flag the chunk if similarity exceeds the threshold
            if similarity > threshold:
                return {
                    "is_duplicate": True,
                    "similarity": similarity,
                    "duplicate_of": existing_chunk["metadata"]["chunk_id"]
                }

        return {
            "is_duplicate": False,
            "similarity": None,
            "duplicate_of": None
        }

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Batch duplicate check failed")
        raise RAGPipelineException(str(e))


# Remove duplicate chunks before indexing
def deduplicate_chunks(
    embedded_chunks,
    threshold=DEDUPLICATION_THRESHOLD,
    user_id=None
):
    try:
        # Validate the input chunks
        validate_embedded_chunks(embedded_chunks)

        # Validate the similarity threshold
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                "Deduplication threshold must be between 0 and 1"
            )

        if not embedded_chunks:
            logger.warning("No chunks received for deduplication")

            return {
                "unique_chunks": [],
                "duplicate_chunks": []
            }

        # Store unique and duplicate chunks separately
        unique_chunks = []
        duplicate_chunks = []

        # Check every incoming chunk
        for chunk in embedded_chunks:

            embedding = chunk["embedding"]

            # Step 1: Check against existing ChromaDB records
            existing_check = check_existing_duplicate(
                embedding=embedding,
                threshold=threshold,
                user_id=user_id
            )

            if existing_check["is_duplicate"]:

                duplicate_info = chunk.copy()
                duplicate_info["duplicate_reason"] = "existing_document"
                duplicate_info["similarity"] = existing_check["similarity"]
                duplicate_info["duplicate_of"] = existing_check["duplicate_of"]

                duplicate_chunks.append(duplicate_info)

                logger.info(
                    f"Duplicate found in ChromaDB: "
                    f"{chunk['metadata']['chunk_id']} | "
                    f"Similarity: {existing_check['similarity']:.4f}"
                )

                continue

            # Step 2: Check against chunks already accepted in this batch
            batch_check = check_batch_duplicate(
                embedding=embedding,
                accepted_chunks=unique_chunks,
                threshold=threshold
            )

            if batch_check["is_duplicate"]:

                duplicate_info = chunk.copy()
                duplicate_info["duplicate_reason"] = "same_batch"
                duplicate_info["similarity"] = batch_check["similarity"]
                duplicate_info["duplicate_of"] = batch_check["duplicate_of"]

                duplicate_chunks.append(duplicate_info)

                logger.info(
                    f"Duplicate found in incoming batch: "
                    f"{chunk['metadata']['chunk_id']} | "
                    f"Similarity: {batch_check['similarity']:.4f}"
                )

                continue

            # Step 3: Preserve the unique chunk
            unique_chunk = chunk.copy()
            unique_chunk["metadata"] = chunk["metadata"].copy()

            unique_chunks.append(unique_chunk)

        logger.info(
            f"Deduplication completed | "
            f"Unique: {len(unique_chunks)} | "
            f"Duplicates: {len(duplicate_chunks)}"
        )

        return {
            "unique_chunks": unique_chunks,
            "duplicate_chunks": duplicate_chunks
        }

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Document deduplication failed")
        raise RAGPipelineException(str(e))


# Test duplicate detection independently
def main():
    try:
        from src.embeddings import generate_embeddings
        from src.vector_store import (
            store_embeddings,
            delete_document
        )

        # Create a sample chunk that already exists in ChromaDB
        existing_chunks = [
            {
                "text": (
                    "Machine learning allows computers "
                    "to learn patterns from data."
                ),
                "metadata": {
                    "source": "dedup_test.txt",
                    "chunk_id": "dedup_existing_001",
                    "chunk_index": 0
                }
            }
        ]

        # Generate embeddings for the existing sample chunk
        existing_embedded = generate_embeddings(existing_chunks)

        # Insert the existing chunk into ChromaDB
        stored_count = store_embeddings(existing_embedded)

        print(f"\nExisting chunks stored: {stored_count}")

        # Create incoming chunks
        incoming_chunks = [
            {
                # Same content as the existing ChromaDB chunk
                "text": (
                    "Machine learning allows computers "
                    "to learn patterns from data."
                ),
                "metadata": {
                    "source": "dedup_test.txt",
                    "chunk_id": "dedup_candidate_001",
                    "chunk_index": 0
                }
            },
            {
                # New and unrelated content
                "text": (
                    "The ocean contains diverse marine life "
                    "and many different ecosystems."
                ),
                "metadata": {
                    "source": "dedup_test.txt",
                    "chunk_id": "dedup_candidate_002",
                    "chunk_index": 1
                }
            },
            {
                # Same content as the second incoming chunk
                "text": (
                    "The ocean contains diverse marine life "
                    "and many different ecosystems."
                ),
                "metadata": {
                    "source": "dedup_test.txt",
                    "chunk_id": "dedup_candidate_003",
                    "chunk_index": 2
                }
            }
        ]

        # Generate embeddings for incoming chunks
        incoming_embedded = generate_embeddings(incoming_chunks)

        # Execute duplicate detection
        results = deduplicate_chunks(
            embedded_chunks=incoming_embedded,
            threshold=DEDUPLICATION_THRESHOLD
        )

        # Display unique chunks
        print("\nUnique Chunks:")

        for chunk in results["unique_chunks"]:
            print("\nChunk ID:", chunk["metadata"]["chunk_id"])
            print("Text:", chunk["text"])

        # Display duplicate chunks
        print("\nDuplicate Chunks:")

        for chunk in results["duplicate_chunks"]:
            print("\nChunk ID:", chunk["metadata"]["chunk_id"])
            print("Duplicate Reason:", chunk["duplicate_reason"])
            print("Similarity:", chunk["similarity"])
            print("Duplicate Of:", chunk["duplicate_of"])

        # Display summary
        print("\nDeduplication Summary:")
        print("Total Incoming Chunks:", len(incoming_embedded))
        print("Unique Chunks:", len(results["unique_chunks"]))
        print("Duplicate Chunks:", len(results["duplicate_chunks"]))

        # Remove only the sample test document
        deleted_count = delete_document(
            source="dedup_test.txt"
        )

        print(f"\nTest chunks deleted: {deleted_count}")

    except RAGPipelineException as e:
        logger.error(f"Deduplication test failed: {e}")
        print(f"Deduplication test failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in deduplication module")
        print(f"Unexpected error: {e}")


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()
