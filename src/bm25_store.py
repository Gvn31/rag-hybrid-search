
import json
import math
import re
import threading

from pathlib import Path

from rank_bm25 import BM25Okapi

from src.logger import logger
from src.exception import RAGPipelineException


# BM25 storage configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BM25_DIR = PROJECT_ROOT / "storage" / "bm25"
BM25_FILE = BM25_DIR / "bm25_index.json"

# Protect read-modify-write operations within this process
INDEX_LOCK = threading.RLock()


# Convert text into lowercase word tokens
def tokenize_text(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        tokens = re.findall(r"\b\w+\b", text.lower())

        return tokens

    except Exception as e:
        logger.exception("Text tokenization failed")
        raise RAGPipelineException(str(e)) from e


# Convert metadata into JSON-compatible values
def prepare_metadata(metadata):
    try:
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        prepared_metadata = {}

        for key, value in metadata.items():

            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "Metadata keys must be non-empty strings"
                )

            if value is None:
                continue

            if isinstance(value, bool):
                prepared_metadata[key] = value

            elif isinstance(value, int):
                prepared_metadata[key] = value

            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Invalid numeric metadata value: {key}"
                    )

                prepared_metadata[key] = value

            elif isinstance(value, str):
                prepared_metadata[key] = value

            else:
                prepared_metadata[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str
                )

        return prepared_metadata

    except Exception as e:
        logger.exception("Metadata preparation failed")
        raise RAGPipelineException(str(e)) from e


# Validate a stored BM25 record
def validate_record(record):
    try:
        if not isinstance(record, dict):
            raise ValueError("Each BM25 record must be a dictionary")

        if not all(
            key in record for key in ["id", "text", "metadata"]
        ):
            raise ValueError(
                "BM25 record must contain id, text and metadata"
            )

        if not isinstance(record["id"], str) or not record["id"].strip():
            raise ValueError("BM25 record ID cannot be empty")

        if not isinstance(record["text"], str) or not record["text"].strip():
            raise ValueError("BM25 record text cannot be empty")

        if not isinstance(record["metadata"], dict):
            raise ValueError("BM25 record metadata must be a dictionary")

        return True

    except Exception as e:
        raise RAGPipelineException(str(e)) from e


# Load every persisted BM25 record

def load_bm25_records():
    try:
        if not BM25_FILE.exists():
            logger.info("No existing BM25 index found")
            return []

        with BM25_FILE.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError("Invalid BM25 index format")

        seen_ids = set()
        for record in records:
            validate_record(record)
            if record["id"] in seen_ids:
                raise ValueError(
                    f"Duplicate BM25 record ID: {record['id']}"
                )
            seen_ids.add(record["id"])

        return records

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to load BM25 records")
        raise RAGPipelineException(str(e)) from e


# Build a searchable BM25 index

def load_bm25_index():
    try:
        records = load_bm25_records()
        searchable_records = []
        searchable_corpus = []

        for record in records:
            tokens = tokenize_text(record["text"])
            if tokens:
                searchable_records.append(record)
                searchable_corpus.append(tokens)

        bm25_index = (
            BM25Okapi(searchable_corpus)
            if searchable_corpus
            else None
        )

        logger.info(
            f"BM25 index loaded with {len(searchable_records)} searchable documents"
        )

        return searchable_records, bm25_index

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to build BM25 index")
        raise RAGPipelineException(str(e)) from e


# Save BM25 document records to persistent storage
def save_bm25_index(records):
    try:
        if not isinstance(records, list):
            raise ValueError("BM25 records must be provided as a list")

        for record in records:
            validate_record(record)

        BM25_DIR.mkdir(parents=True, exist_ok=True)

        temporary_file = BM25_FILE.with_suffix(".tmp")

        with INDEX_LOCK:
            with open(temporary_file, "w", encoding="utf-8") as file:
                json.dump(
                    records,
                    file,
                    ensure_ascii=False,
                    indent=4,
                    allow_nan=False
                )

                file.flush()

            # Replace the previous index after successful writing
            temporary_file.replace(BM25_FILE)

        logger.info(
            f"BM25 index saved successfully: {len(records)} records"
        )

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to save BM25 index")
        raise RAGPipelineException(str(e)) from e


# Add or update document chunks in the BM25 index
def index_chunks(chunks, user_id=None):
    try:
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be provided as a list")

        if not chunks:
            logger.warning("No chunks received for BM25 indexing")
            return 0

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string")

        prepared_chunks = []
        batch_ids = set()

        # Validate the complete batch before updating the index
        for chunk in chunks:

            if not isinstance(chunk, dict):
                raise ValueError("Each chunk must be a dictionary")

            if not all(
                key in chunk for key in ["text", "metadata"]
            ):
                raise ValueError(
                    "Each chunk must contain text and metadata"
                )

            text = chunk["text"]
            metadata = chunk["metadata"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chunk text cannot be empty")

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

            if user_id is not None:
                chunk_metadata["user_id"] = user_id

            storage_id = (
                f"{user_id}_{chunk_id}"
                if user_id is not None
                else chunk_id
            )

            if storage_id in batch_ids:
                raise ValueError(
                    f"Duplicate chunk ID in batch: {storage_id}"
                )

            batch_ids.add(storage_id)

            prepared_chunks.append({
                "id": storage_id,
                "text": text,
                "metadata": chunk_metadata
            })

        # Update existing records without duplicating chunk IDs
        with INDEX_LOCK:
            existing_records = load_bm25_records()

            records_by_id = {
                record["id"]: record
                for record in existing_records
            }

            for record in prepared_chunks:
                records_by_id[record["id"]] = record

            updated_records = list(records_by_id.values())

            save_bm25_index(updated_records)

        logger.info(
            f"Successfully indexed {len(prepared_chunks)} chunks using BM25"
        )

        return len(prepared_chunks)

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("BM25 indexing failed")
        raise RAGPipelineException(str(e)) from e


# Search the BM25 index using keyword matching
def search_bm25(
    query,
    top_k=5,
    user_id=None,
    where=None
):
    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string")

        if where is not None:
            if not isinstance(where, dict) or not where:
                raise ValueError("where must be a non-empty dictionary")

        query_tokens = tokenize_text(query)

        if not query_tokens:
            logger.warning(
                "Search query contains no searchable words"
            )
            return []

        records, _ = load_bm25_index()

        if not records:
            logger.warning("BM25 index is empty")
            return []

        # Apply tenant and metadata filters BEFORE calculating scores
        filtered_records = []

        for record in records:

            metadata = record["metadata"]

            if user_id is not None:
                if metadata.get("user_id") != user_id:
                    continue

            if where is not None:
                matches_filter = True

                for key, condition in where.items():
                    metadata_value = metadata.get(key)

                    # Support ChromaDB-style $in filters.
                    if isinstance(condition, dict):
                        if "$in" in condition:
                            allowed_values = condition["$in"]

                            if not isinstance(allowed_values, (list, tuple, set)):
                                raise ValueError(
                                    f"The $in filter for '{key}' must contain a list or tuple"
                                )

                            if metadata_value not in allowed_values:
                                matches_filter = False
                                break

                        # Support explicit equality filters.
                        elif "$eq" in condition:
                            if metadata_value != condition["$eq"]:
                                matches_filter = False
                                break

                        else:
                            raise ValueError(
                                f"Unsupported BM25 filter operator for '{key}': "
                                f"{list(condition.keys())}"
                            )

                    # Support ordinary exact-match filters.
                    elif metadata_value != condition:
                        matches_filter = False
                        break

                if not matches_filter:
                    continue

            filtered_records.append(record)

        if not filtered_records:
            logger.info("No BM25 records matched the search filters")
            return []

        # Build a search index using only the permitted records
        filtered_corpus = [
            tokenize_text(record["text"])
            for record in filtered_records
        ]

        bm25_index = BM25Okapi(filtered_corpus)

        scores = bm25_index.get_scores(query_tokens)

        ranked_results = []

        for index, record in enumerate(filtered_records):

            ranked_results.append({
                "id": record["id"],
                "text": record["text"],
                "metadata": record["metadata"],
                "score": float(scores[index])
            })

        ranked_results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        search_results = ranked_results[:top_k]

        logger.info(
            f"BM25 search returned {len(search_results)} results"
        )

        return search_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("BM25 search failed")
        raise RAGPipelineException(str(e)) from e


# Delete all indexed chunks belonging to a document
def delete_document(source, user_id=None):
    try:
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "A valid document source is required"
            )

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string")

        with INDEX_LOCK:
            records = load_bm25_records()

            remaining_records = []
            deleted_count = 0

            for record in records:

                metadata = record["metadata"]

                source_matches = metadata.get("source") == source

                user_matches = (
                    user_id is None
                    or metadata.get("user_id") == user_id
                )

                if source_matches and user_matches:
                    deleted_count += 1
                else:
                    remaining_records.append(record)

            if deleted_count == 0:
                logger.warning(
                    f"No BM25 chunks found for document: {source}"
                )
                return 0

            save_bm25_index(remaining_records)

        logger.info(
            f"Deleted {deleted_count} BM25 chunks for {source}"
        )

        return deleted_count

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("BM25 document deletion failed")
        raise RAGPipelineException(str(e)) from e


# Return the total number of indexed document chunks
def get_index_count():
    try:
        records = load_bm25_records()

        total_chunks = len(records)

        logger.info(
            f"Total BM25 indexed chunks: {total_chunks}"
        )

        return total_chunks

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to get BM25 index count")
        raise RAGPipelineException(str(e)) from e


# Test BM25 indexing, search and deletion independently
def main():
    try:
        sample_chunks = [
            {
                "text": (
                    "Artificial intelligence enables machines "
                    "to perform tasks requiring human intelligence."
                ),
                "metadata": {
                    "source": "bm25_test.txt",
                    "chunk_id": "bm25_test_001",
                    "chunk_index": 0
                }
            },
            {
                "text": (
                    "Machine learning allows computers to learn "
                    "patterns from data."
                ),
                "metadata": {
                    "source": "bm25_test.txt",
                    "chunk_id": "bm25_test_002",
                    "chunk_index": 1
                }
            },
            {
                "text": (
                    "Deep learning uses neural networks "
                    "to understand complex data."
                ),
                "metadata": {
                    "source": "bm25_test.txt",
                    "chunk_id": "bm25_test_003",
                    "chunk_index": 2
                }
            }
        ]

        # Index sample chunks
        indexed_count = index_chunks(sample_chunks)

        print(f"\nIndexed chunks: {indexed_count}")

        print(
            "\nTotal indexed chunks:",
            get_index_count()
        )

        # Perform keyword-based search
        query = "machine learning data"

        search_results = search_bm25(
            query=query,
            top_k=3
        )

        print(f"\nSearch query: {query}")
        print("\nBM25 Search Results:")

        for result in search_results:

            print("\nChunk ID:", result["id"])
            print("Text:", result["text"])
            print("BM25 Score:", result["score"])
            print("Metadata:", result["metadata"])

        # Remove only the test document
        deleted_count = delete_document(
            source="bm25_test.txt"
        )

        print(f"\nTest chunks deleted: {deleted_count}")

        print(
            "\nRemaining indexed chunks:",
            get_index_count()
        )

    except RAGPipelineException as e:
        logger.error(f"BM25 test failed: {e}")
        print(f"BM25 test failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in BM25 module")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

