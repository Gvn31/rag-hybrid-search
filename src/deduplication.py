
import math

import numpy as np

from src.logger import logger
from src.exception import RAGPipelineException
from src.vector_store import search_similar_chunks


# Deduplication configuration
DEDUPLICATION_THRESHOLD = 0.95


# Validate the similarity threshold
def validate_threshold(threshold):
    try:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                "Deduplication threshold must be between 0 and 1"
            )

        return float(threshold)

    except Exception as e:
        raise RAGPipelineException(str(e)) from e


# Normalize an embedding vector before similarity comparison
def normalize_embedding(embedding):
    try:
        vector = np.asarray(
            embedding,
            dtype=np.float32
        )

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError(
                "Embedding must be a non-empty 1D vector"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Embedding contains invalid numeric values"
            )

        magnitude = np.linalg.norm(vector)

        if not np.isfinite(magnitude) or magnitude == 0:
            raise ValueError(
                "Embedding magnitude must be finite and non-zero"
            )

        return vector / magnitude

    except Exception as e:
        logger.exception("Embedding normalization failed")
        raise RAGPipelineException(str(e)) from e


# Calculate cosine similarity between two embedding vectors
def calculate_cosine_similarity(embedding_1, embedding_2):
    try:
        vector_1 = normalize_embedding(embedding_1)
        vector_2 = normalize_embedding(embedding_2)

        if vector_1.shape != vector_2.shape:
            raise ValueError(
                "Embedding dimensions do not match"
            )

        similarity = float(
            np.dot(vector_1, vector_2)
        )

        return float(
            np.clip(similarity, -1.0, 1.0)
        )

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Cosine similarity calculation failed")
        raise RAGPipelineException(str(e)) from e


# Validate incoming embedded chunks
def validate_embedded_chunks(embedded_chunks):
    try:
        if not isinstance(embedded_chunks, list):
            raise ValueError(
                "Embedded chunks must be provided as a list"
            )

        seen_ids = set()
        expected_dimension = None

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

            text = chunk["text"]
            metadata = chunk["metadata"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chunk text cannot be empty")

            if not isinstance(metadata, dict):
                raise ValueError(
                    "Chunk metadata must be a dictionary"
                )

            chunk_id = metadata.get("chunk_id")
            source = metadata.get("source")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    "Chunk metadata must contain a valid chunk_id"
                )

            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    "Chunk metadata must contain a valid source"
                )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID in incoming batch: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            vector = normalize_embedding(chunk["embedding"])

            if expected_dimension is None:
                expected_dimension = vector.size

            elif vector.size != expected_dimension:
                raise ValueError(
                    "All embeddings in a batch must have the same dimension"
                )

        return True

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Embedded chunk validation failed")
        raise RAGPipelineException(str(e)) from e


# Check whether a chunk already exists in ChromaDB
def check_existing_duplicate(
    embedding,
    threshold=DEDUPLICATION_THRESHOLD,
    user_id=None
):
    try:
        threshold = validate_threshold(threshold)

        normalized_embedding = normalize_embedding(
            embedding
        ).tolist()

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(
                    "user_id must be a non-empty string"
                )

        # Retrieve the nearest existing vector
        existing_results = search_similar_chunks(
            query_embedding=normalized_embedding,
            top_k=1,
            user_id=user_id
        )

        if not existing_results:
            return {
                "is_duplicate": False,
                "similarity": None,
                "duplicate_of": None
            }

        nearest_chunk = existing_results[0]

        distance = float(nearest_chunk["distance"])

        if not math.isfinite(distance):
            raise ValueError(
                "ChromaDB returned an invalid distance"
            )

        # ChromaDB cosine distance = 1 - cosine similarity
        similarity = float(
            np.clip(1.0 - distance, -1.0, 1.0)
        )

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
        raise RAGPipelineException(str(e)) from e


# Check whether a chunk duplicates another chunk in the same batch
def check_batch_duplicate(
    embedding,
    accepted_chunks,
    threshold=DEDUPLICATION_THRESHOLD
):
    try:
        threshold = validate_threshold(threshold)

        if not isinstance(accepted_chunks, list):
            raise ValueError(
                "Accepted chunks must be provided as a list"
            )

        normalized_embedding = normalize_embedding(
            embedding
        )

        for existing_chunk in accepted_chunks:

            similarity = calculate_cosine_similarity(
                normalized_embedding,
                existing_chunk["embedding"]
            )

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
        raise RAGPipelineException(str(e)) from e


# Remove duplicate chunks before indexing
def deduplicate_chunks(
    embedded_chunks,
    threshold=DEDUPLICATION_THRESHOLD,
    user_id=None
):
    try:
        # Validate all incoming chunks
        validate_embedded_chunks(embedded_chunks)

        threshold = validate_threshold(threshold)

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(
                    "user_id must be a non-empty string"
                )

        if not embedded_chunks:
            logger.warning(
                "No chunks received for deduplication"
            )

            return {
                "unique_chunks": [],
                "duplicate_chunks": []
            }

        unique_chunks = []
        duplicate_chunks = []

        for chunk in embedded_chunks:

            embedding = chunk["embedding"]
            chunk_id = chunk["metadata"]["chunk_id"]

            # Step 1: Check against existing ChromaDB records
            existing_check = check_existing_duplicate(
                embedding=embedding,
                threshold=threshold,
                user_id=user_id
            )

            if existing_check["is_duplicate"]:

                duplicate_info = chunk.copy()
                duplicate_info["metadata"] = chunk["metadata"].copy()

                duplicate_info["duplicate_reason"] = "existing_document"
                duplicate_info["similarity"] = existing_check["similarity"]
                duplicate_info["duplicate_of"] = existing_check["duplicate_of"]

                duplicate_chunks.append(duplicate_info)

                logger.info(
                    f"Duplicate found in ChromaDB: {chunk_id} | "
                    f"Similarity: {existing_check['similarity']:.4f}"
                )

                continue

            # Step 2: Check against previously accepted batch chunks
            batch_check = check_batch_duplicate(
                embedding=embedding,
                accepted_chunks=unique_chunks,
                threshold=threshold
            )

            if batch_check["is_duplicate"]:

                duplicate_info = chunk.copy()
                duplicate_info["metadata"] = chunk["metadata"].copy()

                duplicate_info["duplicate_reason"] = "same_batch"
                duplicate_info["similarity"] = batch_check["similarity"]
                duplicate_info["duplicate_of"] = batch_check["duplicate_of"]

                duplicate_chunks.append(duplicate_info)

                logger.info(
                    f"Duplicate found in incoming batch: {chunk_id} | "
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
        raise RAGPipelineException(str(e)) from e


# Test duplicate detection independently
def main():
    try:
        from src.embeddings import generate_embeddings
        from src.vector_store import (
            store_embeddings,
            delete_document
        )

        # Prepare sample document texts
        existing_text = (
            "Machine learning allows computers "
            "to learn patterns from data."
        )

        incoming_texts = [
            existing_text,
            (
                "The ocean contains diverse marine life "
                "and many different ecosystems."
            ),
            (
                "The ocean contains diverse marine life "
                "and many different ecosystems."
            )
        ]

        # Create sample chunks with required metadata
        def create_sample_chunk(text, chunk_id, chunk_index):
            return {
                "text": text,
                "metadata": {
                    "source": "dedup_test.txt",
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "chunking_strategy": "test",
                    "char_count": len(text),
                    "start_index": 0,
                    "end_index": len(text),
                    "section_heading": None
                }
            }

        existing_chunks = [
            create_sample_chunk(
                existing_text,
                "dedup_existing_001",
                0
            )
        ]

        incoming_chunks = [
            create_sample_chunk(
                incoming_texts[0],
                "dedup_candidate_001",
                0
            ),
            create_sample_chunk(
                incoming_texts[1],
                "dedup_candidate_002",
                1
            ),
            create_sample_chunk(
                incoming_texts[2],
                "dedup_candidate_003",
                2
            )
        ]

        # Generate and store an existing embedding
        existing_embedded = generate_embeddings(
            existing_chunks
        )

        stored_count = store_embeddings(
            existing_embedded
        )

        print(f"\nExisting chunks stored: {stored_count}")

        # Generate incoming embeddings
        incoming_embedded = generate_embeddings(
            incoming_chunks
        )

        # Perform duplicate detection
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

        # Remove only the test document
        deleted_count = delete_document(
            source="dedup_test.txt"
        )

        print(f"\nTest chunks deleted: {deleted_count}")

    except RAGPipelineException as e:
        logger.error(f"Deduplication test failed: {e}")
        print(f"Deduplication test failed: {e}")

    except Exception as e:
        logger.exception(
            "Unexpected error in deduplication module"
        )
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

