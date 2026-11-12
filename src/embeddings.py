
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
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            logger.info("Embedding model loaded successfully")

        return _embedding_model

    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise RAGPipelineException(str(e))


# Generate vector embeddings for document chunks
def generate_embeddings(chunks):
    try:
        # Validate the input
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be provided as a list")

        if not chunks:
            logger.warning("No chunks received for embedding generation")
            return []

        texts = []

        # Validate each chunk before generating embeddings
        for chunk in chunks:

            if not isinstance(chunk, dict):
                raise ValueError("Each chunk must be a dictionary")

            if "text" not in chunk or "metadata" not in chunk:
                raise ValueError(
                    "Each chunk must contain text and metadata"
                )

            if not isinstance(chunk["text"], str):
                raise ValueError("Chunk text must be a string")

            if not isinstance(chunk["metadata"], dict):
                raise ValueError("Chunk metadata must be a dictionary")

            if not chunk["text"].strip():
                raise ValueError("Chunk text cannot be empty")

            texts.append(chunk["text"])

        # Load the embedding model
        model = get_embedding_model()

        # Convert all chunk texts into vector embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Attach each generated embedding to its original chunk
        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):

            # Preserve the original chunk metadata
            chunk_metadata = chunk["metadata"].copy()

            # Store embedding-related information
            chunk_metadata.update({
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": int(len(embedding))
            })

            embedded_chunks.append({
                "text": chunk["text"],
                "metadata": chunk_metadata,
                "embedding": embedding.tolist()
            })

        logger.info(
            f"Generated embeddings for {len(embedded_chunks)} chunks"
        )

        return embedded_chunks

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Embedding generation failed")
        raise RAGPipelineException(str(e))


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
                    "chunk_index": 0
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
                    "chunk_index": 1
                }
            }
        ]

        # Generate embeddings for the sample chunks
        embedded_chunks = generate_embeddings(sample_chunks)

        # Display the generated embedding information
        for chunk in embedded_chunks:

            print("\nChunk ID:", chunk["metadata"]["chunk_id"])
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
                "Metadata:",
                chunk["metadata"]
            )

        print(
            f"\nTotal embeddings generated: {len(embedded_chunks)}"
        )

    except RAGPipelineException as e:
        logger.error(f"Embedding generation failed: {e}")
        print(f"Embedding generation failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in embeddings module")
        print(f"Unexpected error: {e}")


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()
