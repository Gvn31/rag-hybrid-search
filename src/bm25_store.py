
import json
import re

from pathlib import Path

from rank_bm25 import BM25Okapi

from src.logger import logger
from src.exception import RAGPipelineException


# Define the persistent BM25 storage directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BM25_DIR = PROJECT_ROOT / "storage" / "bm25"
BM25_FILE = BM25_DIR / "bm25_index.json"


# Convert text into lowercase word tokens
def tokenize_text(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        # Extract words while preserving Unicode characters
        tokens = re.findall(r"\b\w+\b", text.lower())

        return tokens

    except Exception as e:
        logger.exception("Text tokenization failed")
        raise RAGPipelineException(str(e))


# Convert metadata into JSON-compatible scalar values
def prepare_metadata(metadata):
    try:
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        prepared_metadata = {}

        for key, value in metadata.items():

            # Skip missing metadata values
            if value is None:
                continue

            # Preserve supported scalar values
            if isinstance(value, (str, int, float, bool)):
                prepared_metadata[key] = value

            else:
                # Convert nested values into strings
                prepared_metadata[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str
                )

        return prepared_metadata

    except Exception as e:
        logger.exception("Metadata preparation failed")
        raise RAGPipelineException(str(e))


# Load the saved BM25 documents and build the search index
def load_bm25_index():
    try:
        # Return an empty index when no saved index exists
        if not BM25_FILE.exists():
            logger.info("No existing BM25 index found")

            return [], None

        # Read the saved document records
        with open(BM25_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError("Invalid BM25 index format")

        # Tokenize all stored document chunks
        corpus = [
            tokenize_text(record["text"])
            for record in records
        ]

        # Remove records that contain no searchable words
        valid_records = []
        valid_corpus = []

        for record, tokens in zip(records, corpus):
            if tokens:
                valid_records.append(record)
                valid_corpus.append(tokens)

        # Build the BM25 index
        if valid_corpus:
            bm25_index = BM25Okapi(valid_corpus)
        else:
            bm25_index = None

        logger.info(
            f"BM25 index loaded with {len(valid_records)} documents"
        )

        return valid_records, bm25_index

    except Exception as e:
        logger.exception("Failed to load BM25 index")
        raise RAGPipelineException(str(e))


# Save BM25 document records to persistent storage
def save_bm25_index(records):
    try:
        # Create the storage directory if it does not exist
        BM25_DIR.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file before replacing the index
        temporary_file = BM25_FILE.with_suffix(".tmp")

        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                ensure_ascii=False,
                indent=4
            )

        # Replace the previous index with the updated file
        temporary_file.replace(BM25_FILE)

        logger.info(
            f"BM25 index saved successfully: {len(records)} documents"
        )

    except Exception as e:
        logger.exception("Failed to save BM25 index")
        raise RAGPipelineException(str(e))


# Add or update document chunks in the BM25 index
def index_chunks(chunks, user_id=None):
    try:
        # Validate the input
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be provided as a list")

        if not chunks:
            logger.warning("No chunks received for BM25 indexing")
            return 0

        # Load existing document records
        existing_records, _ = load_bm25_index()

        # Create a dictionary for efficient upsert operations
        records_by_id = {
            record["id"]: record
            for record in existing_records
        }

        # Process each document chunk
        for chunk in chunks:

            if not isinstance(chunk, dict):
                raise ValueError("Each chunk must be a dictionary")

            if "text" not in chunk or "metadata" not in chunk:
                raise ValueError(
                    "Each chunk must contain text and metadata"
                )

            text = chunk["text"]
            metadata = chunk["metadata"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Chunk text cannot be empty")

            if not isinstance(metadata, dict):
                raise ValueError("Chunk metadata must be a dictionary")

            # Prepare chunk metadata
            chunk_metadata = prepare_metadata(metadata)

            # Require a unique chunk identifier
            chunk_id = chunk_metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Chunk metadata must contain chunk_id"
                )

            # Add user ID for multi-user document isolation
            if user_id is not None:
                chunk_metadata["user_id"] = str(user_id)

            # Generate a user-specific storage identifier
            storage_id = (
                f"{user_id}_{chunk_id}"
                if user_id is not None
                else str(chunk_id)
            )

            # Store the document text and metadata
            records_by_id[storage_id] = {
                "id": storage_id,
                "text": text,
                "metadata": chunk_metadata
            }

        # Convert records back into a list
        updated_records = list(records_by_id.values())

        # Persist the updated index
        save_bm25_index(updated_records)

        logger.info(
            f"Successfully indexed {len(chunks)} chunks using BM25"
        )

        return len(chunks)

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("BM25 indexing failed")
        raise RAGPipelineException(str(e))


# Search the BM25 index using keyword matching
def search_bm25(
    query,
    top_k=5,
    user_id=None,
    where=None
):
    try:
        # Validate the search query
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if where is not None and not isinstance(where, dict):
            raise ValueError("where must be a dictionary")

        # Convert the query into tokens
        query_tokens = tokenize_text(query)

        if not query_tokens:
            logger.warning("Search query contains no searchable words")
            return []

        # Load the current BM25 index
        records, bm25_index = load_bm25_index()

        if not records or bm25_index is None:
            logger.warning("BM25 index is empty")
            return []

        # Calculate relevance scores for all documents
        scores = bm25_index.get_scores(query_tokens)

        ranked_results = []

        # Apply metadata and user filters before returning results
        for index, record in enumerate(records):

            metadata = record["metadata"]

            # Restrict results to the requested user
            if user_id is not None:
                if metadata.get("user_id") != str(user_id):
                    continue

            # Apply additional exact-match metadata filters
            if where is not None:
                matches_filter = all(
                    metadata.get(key) == value
                    for key, value in where.items()
                )

                if not matches_filter:
                    continue

            ranked_results.append({
                "id": record["id"],
                "text": record["text"],
                "metadata": metadata,
                "score": float(scores[index])
            })

        # Rank matching documents by BM25 score
        ranked_results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        # Return only the requested number of results
        search_results = ranked_results[:top_k]

        logger.info(
            f"BM25 search returned {len(search_results)} results"
        )

        return search_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("BM25 search failed")
        raise RAGPipelineException(str(e))


# Delete all indexed chunks belonging to a document
def delete_document(source, user_id=None):
    try:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A valid document source is required")

        # Load existing document records
        records, _ = load_bm25_index()

        # Identify chunks that should be retained
        remaining_records = []

        deleted_count = 0

        for record in records:

            metadata = record["metadata"]

            # Check whether the chunk belongs to the requested source
            source_matches = metadata.get("source") == source

            # Check the user ID when supplied
            user_matches = (
                user_id is None
                or metadata.get("user_id") == str(user_id)
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

        # Save the index without the deleted document chunks
        save_bm25_index(remaining_records)

        logger.info(
            f"Deleted {deleted_count} BM25 chunks for {source}"
        )

        return deleted_count

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("BM25 document deletion failed")
        raise RAGPipelineException(str(e))


# Return the total number of indexed document chunks
def get_index_count():
    try:
        records, _ = load_bm25_index()

        total_chunks = len(records)

        logger.info(f"Total BM25 indexed chunks: {total_chunks}")

        return total_chunks

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to get BM25 index count")
        raise RAGPipelineException(str(e))


# Test BM25 indexing, search and deletion independently
def main():
    try:
        # Create sample document chunks
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

        # Add sample chunks to the BM25 index
        indexed_count = index_chunks(sample_chunks)

        print(f"\nIndexed chunks: {indexed_count}")

        # Display the total number of indexed chunks
        print(
            "\nTotal indexed chunks:",
            get_index_count()
        )

        # Perform a keyword-based search
        query = "machine learning data"

        search_results = search_bm25(
            query=query,
            top_k=3
        )

        # Display the retrieved search results
        print(f"\nSearch query: {query}")
        print("\nBM25 Search Results:")

        for result in search_results:
            print("\nChunk ID:", result["id"])
            print("Text:", result["text"])
            print("BM25 Score:", result["score"])
            print("Metadata:", result["metadata"])

        # Remove only the sample test document
        deleted_count = delete_document(
            source="bm25_test.txt"
        )

        print(f"\nTest chunks deleted: {deleted_count}")

        # Display the remaining index count
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


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()
