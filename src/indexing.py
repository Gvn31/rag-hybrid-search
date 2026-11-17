
import math

from src.logger import logger
from src.exception import RAGPipelineException

from src.deduplication import (
    deduplicate_chunks,
    DEDUPLICATION_THRESHOLD,
    validate_threshold
)

from src.vector_store import (
    store_embeddings,
    delete_document as delete_vector_document,
    initialize_vector_store
)

from src.bm25_store import (
    index_chunks,
    delete_document as delete_bm25_document,
    load_bm25_index
)


# Validate embedded chunks before indexing
def validate_chunks(embedded_chunks):
    try:
        if not isinstance(embedded_chunks, list):
            raise ValueError("Embedded chunks must be a list")

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
            embedding = chunk["embedding"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chunk text cannot be empty")

            if not isinstance(metadata, dict):
                raise ValueError(
                    "Chunk metadata must be a dictionary"
                )

            source = metadata.get("source")
            chunk_id = metadata.get("chunk_id")

            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    "Chunk metadata must contain a valid source"
                )

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    "Chunk metadata must contain a valid chunk_id"
                )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID in batch: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            if not isinstance(embedding, (list, tuple)) or not embedding:
                raise ValueError(
                    "Embedding must be a non-empty list or tuple"
                )

            vector = []

            for value in embedding:

                if isinstance(value, bool):
                    raise ValueError(
                        "Embedding cannot contain boolean values"
                    )

                numeric_value = float(value)

                if not math.isfinite(numeric_value):
                    raise ValueError(
                        "Embedding contains invalid numeric values"
                    )

                vector.append(numeric_value)

            if expected_dimension is None:
                expected_dimension = len(vector)

            elif len(vector) != expected_dimension:
                raise ValueError(
                    "All embeddings must have the same dimension"
                )

        return True

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Indexing input validation failed")
        raise RAGPipelineException(str(e)) from e


# Validate that all chunks belong to one document
def validate_document_source(embedded_chunks):
    try:
        if not isinstance(embedded_chunks, list):
            raise ValueError("Embedded chunks must be a list")

        sources = {
            chunk["metadata"]["source"]
            for chunk in embedded_chunks
        }

        if len(sources) > 1:
            raise ValueError(
                "All chunks in one indexing operation "
                "must belong to the same document source"
            )

        if not sources:
            return None

        return next(iter(sources))

    except Exception as e:
        logger.exception("Document source validation failed")
        raise RAGPipelineException(str(e)) from e


# Build a ChromaDB metadata filter
def build_metadata_filter(source, user_id=None):
    try:
        filters = []

        if source is not None:
            filters.append({"source": source})

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(
                    "user_id must be a non-empty string"
                )

            filters.append({"user_id": user_id})

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    except Exception as e:
        logger.exception("Metadata filter creation failed")
        raise RAGPipelineException(str(e)) from e


# Retrieve chunk IDs belonging to a document in ChromaDB
def get_chroma_chunk_ids(source, user_id=None):
    try:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A valid source is required")

        _, collection = initialize_vector_store()

        where_filter = build_metadata_filter(
            source=source,
            user_id=user_id
        )

        results = collection.get(
            where=where_filter,
            include=["metadatas"]
        )

        chunk_ids = {
            metadata["chunk_id"]
            for metadata in results.get("metadatas", [])
            if metadata and metadata.get("chunk_id")
        }

        return chunk_ids

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to retrieve ChromaDB chunk IDs")
        raise RAGPipelineException(str(e)) from e


# Retrieve chunk IDs belonging to a document in BM25
def get_bm25_chunk_ids(source, user_id=None):
    try:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A valid source is required")

        records, _ = load_bm25_index()

        chunk_ids = set()

        for record in records:

            metadata = record.get("metadata", {})

            source_matches = metadata.get("source") == source

            user_matches = (
                user_id is None
                or metadata.get("user_id") == user_id
            )

            if source_matches and user_matches:
                chunk_id = metadata.get("chunk_id")

                if chunk_id:
                    chunk_ids.add(chunk_id)

        return chunk_ids

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to retrieve BM25 chunk IDs")
        raise RAGPipelineException(str(e)) from e


# Verify that both indexes contain matching chunk IDs
def verify_index_consistency(
    source,
    expected_chunk_ids=None,
    user_id=None
):
    try:
        chroma_ids = get_chroma_chunk_ids(
            source=source,
            user_id=user_id
        )

        bm25_ids = get_bm25_chunk_ids(
            source=source,
            user_id=user_id
        )

        matching_ids = chroma_ids == bm25_ids

        expected_match = (
            expected_chunk_ids is None
            or (
                chroma_ids == set(expected_chunk_ids)
                and bm25_ids == set(expected_chunk_ids)
            )
        )

        verification_result = {
            "source": source,
            "chroma_chunk_ids": sorted(chroma_ids),
            "bm25_chunk_ids": sorted(bm25_ids),
            "matching_ids": matching_ids,
            "expected_ids_match": expected_match,
            "status": (
                "success"
                if matching_ids and expected_match
                else "mismatch"
            )
        }

        if verification_result["status"] != "success":
            logger.error(
                f"Index consistency verification failed for {source}"
            )
        else:
            logger.info(
                f"Index consistency verified for {source}"
            )

        return verification_result

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Index consistency verification failed")
        raise RAGPipelineException(str(e)) from e


# Delete a document from both indexes
def delete_document_from_indexes(source, user_id=None):
    try:
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "A valid document source is required"
            )

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(
                    "user_id must be a non-empty string"
                )

        logger.info(
            f"Starting synchronized deletion for document: {source}"
        )

        chroma_error = None
        bm25_error = None

        chroma_deleted = 0
        bm25_deleted = 0

        # Delete from ChromaDB
        try:
            chroma_deleted = delete_vector_document(
                source=source,
                user_id=user_id
            )

        except Exception as e:
            chroma_error = str(e)
            logger.exception("ChromaDB document deletion failed")

        # Delete from BM25 independently
        try:
            bm25_deleted = delete_bm25_document(
                source=source,
                user_id=user_id
            )

        except Exception as e:
            bm25_error = str(e)
            logger.exception("BM25 document deletion failed")

        # Verify deletion from both indexes
        verification = verify_index_consistency(
            source=source,
            expected_chunk_ids=[],
            user_id=user_id
        )

        deletion_summary = {
            "source": source,
            "chroma_deleted": chroma_deleted,
            "bm25_deleted": bm25_deleted,
            "verification": verification,
            "status": "success"
        }

        if chroma_error or bm25_error:
            deletion_summary["status"] = "partial_failure"
            deletion_summary["chroma_error"] = chroma_error
            deletion_summary["bm25_error"] = bm25_error

            raise RAGPipelineException(
                f"Document deletion was incomplete: "
                f"ChromaDB error={chroma_error}, "
                f"BM25 error={bm25_error}"
            )

        if verification["status"] != "success":
            deletion_summary["status"] = "verification_failed"

            raise RAGPipelineException(
                "Document deletion completed but verification failed"
            )

        logger.info(
            f"Synchronized deletion completed for {source} | "
            f"ChromaDB: {chroma_deleted} | BM25: {bm25_deleted}"
        )

        return deletion_summary

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Synchronized document deletion failed")
        raise RAGPipelineException(str(e)) from e


# Index a document in ChromaDB and BM25
def index_embedded_chunks(
    embedded_chunks,
    user_id=None,
    threshold=DEDUPLICATION_THRESHOLD,
    replace_document=True
):
    try:
        # Step 1: Validate everything before modifying existing data
        validate_chunks(embedded_chunks)

        if not isinstance(replace_document, bool):
            raise ValueError(
                "replace_document must be a boolean"
            )

        threshold = validate_threshold(threshold)

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(
                    "user_id must be a non-empty string"
                )

        if not embedded_chunks:
            logger.warning("No chunks received for indexing")

            return {
                "total_chunks": 0,
                "unique_chunks": 0,
                "duplicate_chunks": 0,
                "chroma_indexed": 0,
                "bm25_indexed": 0,
                "verification": None,
                "status": "empty"
            }

        source = validate_document_source(embedded_chunks)

        logger.info(
            f"Starting synchronized indexing for {source}"
        )

        indexing_summary = {
            "source": source,
            "total_chunks": len(embedded_chunks),
            "unique_chunks": 0,
            "duplicate_chunks": 0,
            "chroma_indexed": 0,
            "bm25_indexed": 0,
            "verification": None,
            "status": "pending"
        }

        # Step 2: Replace the previous document version if requested
        if replace_document:

            logger.info(
                f"Replacing previous document version: {source}"
            )

            delete_document_from_indexes(
                source=source,
                user_id=user_id
            )

        # Step 3: Identify duplicate and unique chunks
        deduplication_results = deduplicate_chunks(
            embedded_chunks=embedded_chunks,
            threshold=threshold,
            user_id=user_id
        )

        unique_chunks = deduplication_results["unique_chunks"]
        duplicate_chunks = deduplication_results["duplicate_chunks"]

        indexing_summary["unique_chunks"] = len(unique_chunks)
        indexing_summary["duplicate_chunks"] = len(duplicate_chunks)

        # Stop when no unique chunks remain
        if not unique_chunks:

            indexing_summary["status"] = "all_duplicates"

            logger.info(
                f"No unique chunks available for indexing: {source}"
            )

            return indexing_summary

        # Step 4: Store unique chunks in ChromaDB
        logger.info("Storing unique chunks in ChromaDB")

        chroma_count = store_embeddings(
            embedded_chunks=unique_chunks,
            user_id=user_id
        )

        indexing_summary["chroma_indexed"] = chroma_count

        # Step 5: Index the same chunks in BM25
        logger.info("Indexing unique chunks in BM25")

        try:
            bm25_count = index_chunks(
                chunks=unique_chunks,
                user_id=user_id
            )

            indexing_summary["bm25_indexed"] = bm25_count

        except Exception as e:
            indexing_summary["status"] = "partial_failure"

            logger.critical(
                f"ChromaDB indexing succeeded but BM25 failed for {source}. "
                "Manual reconciliation or retry is required."
            )

            raise RAGPipelineException(
                f"Partial indexing failure for {source}. "
                f"ChromaDB stored {chroma_count} chunks. "
                f"BM25 indexing failed: {str(e)}"
            ) from e

        # Step 6: Compare indexed counts
        if chroma_count != bm25_count:
            indexing_summary["status"] = "count_mismatch"

            raise RAGPipelineException(
                f"Index count mismatch: "
                f"ChromaDB={chroma_count}, BM25={bm25_count}"
            )

        # Step 7: Verify the actual chunk IDs in both indexes
        expected_chunk_ids = {
            chunk["metadata"]["chunk_id"]
            for chunk in unique_chunks
        }

        verification = verify_index_consistency(
            source=source,
            expected_chunk_ids=expected_chunk_ids,
            user_id=user_id
        )

        indexing_summary["verification"] = verification

        if verification["status"] != "success":
            indexing_summary["status"] = "verification_failed"

            raise RAGPipelineException(
                f"ChromaDB and BM25 chunk IDs do not match "
                f"for document {source}"
            )

        # Step 8: Mark indexing as successful
        indexing_summary["status"] = "success"

        logger.info(
            f"Synchronized indexing completed for {source} | "
            f"Unique: {len(unique_chunks)} | "
            f"Duplicates: {len(duplicate_chunks)}"
        )

        return indexing_summary

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Synchronized indexing failed")
        raise RAGPipelineException(str(e)) from e


# Test synchronized indexing, re-indexing and deletion
def main():
    try:
        from src.embeddings import generate_embeddings

        # Sample document texts
        sample_texts = [
            (
                "Artificial intelligence enables machines "
                "to perform tasks requiring human intelligence."
            ),
            (
                "Machine learning allows computers to learn "
                "patterns from data."
            ),
            (
                "Machine learning allows computers to learn "
                "patterns from data."
            )
        ]

        sample_chunks = []

        # Create chunks with metadata required by embeddings.py
        current_position = 0

        for index, text in enumerate(sample_texts):

            start_index = current_position
            end_index = start_index + len(text)

            sample_chunks.append({
                "text": text,
                "metadata": {
                    "source": "indexing_test.txt",
                    "chunk_id": f"indexing_test_{index + 1:03d}",
                    "chunk_index": index,
                    "chunking_strategy": "test",
                    "char_count": len(text),
                    "start_index": start_index,
                    "end_index": end_index,
                    "section_heading": None
                }
            })

            current_position = end_index + 1

        # Generate embeddings
        embedded_chunks = generate_embeddings(sample_chunks)

        # Step 1: Initial indexing
        indexing_results = index_embedded_chunks(
            embedded_chunks=embedded_chunks,
            replace_document=True
        )

        print("\nInitial Indexing Results:")

        for key, value in indexing_results.items():
            print(f"{key}: {value}")

        # Step 2: Verify indexed document
        verification = verify_index_consistency(
            source="indexing_test.txt"
        )

        print("\nIndex Consistency Verification:")

        for key, value in verification.items():
            print(f"{key}: {value}")

        # Step 3: Re-index the same document
        reindexing_results = index_embedded_chunks(
            embedded_chunks=embedded_chunks,
            replace_document=True
        )

        print("\nRe-indexing Results:")

        for key, value in reindexing_results.items():
            print(f"{key}: {value}")

        # Step 4: Delete the test document from both indexes
        deletion_results = delete_document_from_indexes(
            source="indexing_test.txt"
        )

        print("\nSynchronized Deletion Results:")

        for key, value in deletion_results.items():
            print(f"{key}: {value}")

    except RAGPipelineException as e:
        logger.error(f"Indexing test failed: {e}")
        print(f"Indexing test failed: {e}")

    except Exception as e:
        logger.exception(
            "Unexpected error in indexing module"
        )
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
