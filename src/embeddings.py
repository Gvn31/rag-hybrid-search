
import sys
import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.logger import logger
from src.exception import RAGPipelineException


# Store the model so it is loaded only once in this module
_embedding_model = None


# Load the sentence transformer embedding model
def get_embedding_model():

    global _embedding_model

    try:
        if _embedding_model is None:

            logger.info(
                f"Loading embedding model: {EMBEDDING_MODEL}"
            )

            _embedding_model = SentenceTransformer(
                EMBEDDING_MODEL
            )

            logger.info(
                "Embedding model loaded successfully"
            )

        return _embedding_model

    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise RAGPipelineException(str(e)) from e


# Validate chunks before generating embeddings
def validate_chunks(chunks):

    try:
        if not isinstance(chunks, list):
            raise TypeError(
                "Chunks must be provided as a list"
            )

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Chunk at index {index} must be a dictionary"
                )

            if "text" not in chunk or "metadata" not in chunk:
                raise ValueError(
                    f"Chunk at index {index} must contain "
                    "text and metadata"
                )

            if not isinstance(chunk["text"], str):
                raise TypeError(
                    f"Chunk text at index {index} must be a string"
                )

            if not chunk["text"].strip():
                raise ValueError(
                    f"Chunk text at index {index} cannot be empty"
                )

            if not isinstance(chunk["metadata"], dict):
                raise TypeError(
                    f"Chunk metadata at index {index} "
                    "must be a dictionary"
                )

            # Validate required metadata
            required_metadata = [
                "source",
                "chunk_id",
                "chunk_index",
                "chunking_strategy",
                "char_count",
                "start_index",
                "end_index"
            ]

            missing_fields = [
                field
                for field in required_metadata
                if field not in chunk["metadata"]
            ]

            if missing_fields:
                raise ValueError(
                    f"Missing metadata at chunk {index}: "
                    f"{missing_fields}"
                )

            if chunk["metadata"]["char_count"] != len(chunk["text"]):
                raise ValueError(
                    f"Character count mismatch at chunk {index}"
                )

            if (
                chunk["metadata"]["start_index"] < 0
                or chunk["metadata"]["end_index"]
                < chunk["metadata"]["start_index"]
            ):
                raise ValueError(
                    f"Invalid character positions at chunk {index}"
                )

        return True

    except Exception as e:
        logger.exception("Chunk validation failed")
        raise RAGPipelineException(str(e)) from e


# Generate vector embeddings for document chunks
def generate_embeddings(chunks):

    try:
        # Validate input chunks
        validate_chunks(chunks)

        if not chunks:
            logger.warning(
                "No chunks received for embedding generation"
            )
            return []

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # Load the embedding model
        model = get_embedding_model()

        # Generate normalized vector embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Convert output into a NumPy array
        embeddings = np.asarray(embeddings)

        # Validate embedding output dimensions
        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding output must be a 2D array"
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Number of generated embeddings does not "
                "match the number of input chunks"
            )

        if embeddings.shape[1] == 0:
            raise ValueError(
                "Embedding dimension cannot be zero"
            )

        # Validate numerical values
        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Generated embeddings contain NaN or Infinity values"
            )

        # Attach embeddings to their original chunks
        embedded_chunks = []

        for index, chunk in enumerate(chunks):

            embedding = embeddings[index]

            # Preserve the original chunk metadata
            chunk_metadata = chunk["metadata"].copy()

            # Store embedding-related information
            chunk_metadata.update({
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": int(embeddings.shape[1])
            })

            embedded_chunks.append({
                "text": chunk["text"],
                "metadata": chunk_metadata,
                "embedding": embedding.astype(float).tolist()
            })

        logger.info(
            f"Generated embeddings for "
            f"{len(embedded_chunks)} chunks | "
            f"Dimension: {embeddings.shape[1]}"
        )

        return embedded_chunks

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Embedding generation failed")
        raise RAGPipelineException(str(e)) from e


# Test embedding generation independently
def main():

    try:
        # Sample chunks for testing
        sample_chunks = [
            {
                "text": (
                    "Artificial intelligence enables machines "
                    "to perform tasks that require human intelligence."
                ),
                "metadata": {
                    "source": "sample_document.txt",
                    "chunk_id": "chunk_001",
                    "chunk_index": 0,
                    "chunking_strategy": "recursive",
                    "section_heading": "Introduction",
                    "char_count":len(
                    "Artificial intelligence enables machines "
                    "to perform tasks that require human intelligence."
                    ),
                    "start_index": 0,
                    "end_index": 91
                }
            },
            {
                "text": (
                    "Machine learning allows computers to learn "
                    "patterns from data."
                ),
                "metadata": {
                    "source": "sample_document.txt",
                    "chunk_id": "chunk_002",
                    "chunk_index": 1,
                    "chunking_strategy": "recursive",
                    "section_heading": "Machine Learning",
                    "char_count": len(
                        "Machine learning allows computers to learn "
                        "patterns from data."
                    ),
                    "start_index": 92,
                    "end_index": 161
                }
            }
        ]

        # Generate embeddings for the sample chunks
        embedded_chunks = generate_embeddings(sample_chunks)

        # Display the generated embedding information
        for chunk in embedded_chunks:

            print(
                "\nChunk ID:",
                chunk["metadata"]["chunk_id"]
            )

            print("Text:", chunk["text"])

            print(
                "Embedding Dimension:",
                len(chunk["embedding"])
            )

            print(
                "First 10 Vector Values:",
                chunk["embedding"][:10]
            )

            print(
                "Embedding Model:",
                chunk["metadata"]["embedding_model"]
            )

            print(
                "Metadata:",
                chunk["metadata"]
            )

        print(
            f"\nTotal embeddings generated: "
            f"{len(embedded_chunks)}"
        )

    except RAGPipelineException as e:
        logger.error(
            f"Embedding generation failed: {e}"
        )
        print(f"Embedding generation failed: {e}")

    except Exception as e:
        logger.exception(
            "Unexpected error in embeddings module"
        )
        print(f"Unexpected error: {e}")


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()
