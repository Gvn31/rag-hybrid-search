import sys

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, RERANK_TOP_K
from src.logger import logger
from src.exception import RAGPipelineException


# Balance cross-encoder relevance with the hybrid search ranking
RERANK_RRF_K = 60
CROSS_ENCODER_RANK_WEIGHT = 0.45
HYBRID_RANK_WEIGHT = 0.55


# Store the Cross-Encoder model
_reranker_model = None


# Load the Cross-Encoder model only when required
def get_reranker_model():
    global _reranker_model

    try:
        if _reranker_model is None:
            logger.info(
                f"Loading the reranker model: {RERANKER_MODEL}"
            )

            _reranker_model = CrossEncoder(
                RERANKER_MODEL,
                max_length=512
            )

            logger.info("Cross-Encoder model loaded successfully")

        return _reranker_model

    except Exception as e:
        logger.exception("Failed to load the reranker model")
        raise RAGPipelineException(str(e))


# Validate query, retrieved chunks, and top_k
def validate_input(query, retrieved_chunks, top_k):

    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string"
            )

        if not isinstance(retrieved_chunks, list):
            raise ValueError(
                "Retrieved chunks must be a list"
            )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer"
            )

        chunk_ids = set()

        for chunk in retrieved_chunks:

            if not isinstance(chunk, dict):
                raise ValueError(
                    "Each retrieved chunk must be a dictionary"
                )

            chunk_id = chunk.get("id")

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    "Each chunk must contain a valid ID"
                )

            if chunk_id in chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID found: {chunk_id}"
                )

            chunk_ids.add(chunk_id)

            text = chunk.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Chunk {chunk_id} must contain valid text"
                )

            metadata = chunk.get("metadata")

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Chunk {chunk_id} must contain valid metadata"
                )

        return True

    except Exception as e:
        logger.exception("Reranking input validation failed")
        raise RAGPipelineException(str(e))


# Calculate relevance scores for query-document pairs
def calculate_relevance_scores(query, retrieved_chunks):

    try:
        if not retrieved_chunks:
            return []

        # Load the Cross-Encoder model
        model = get_reranker_model()

        # Create query-document pairs
        sentence_pairs = [
            (query.strip(), chunk["text"])
            for chunk in retrieved_chunks
        ]

        # Predict relevance scores
        scores = model.predict(
            sentence_pairs,
            batch_size=16,
            show_progress_bar=False
        )

        # Convert model output into a consistent array
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        # Validate score count
        if len(scores) != len(retrieved_chunks):
            raise ValueError(
                "Relevance score count does not match chunk count"
            )

        # Validate score values
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Cross-Encoder generated invalid relevance scores"
            )

        logger.info(
            f"Calculated relevance scores for {len(scores)} chunks"
        )

        return scores.tolist()

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception(
            "Failed to calculate relevance scores"
        )
        raise RAGPipelineException(str(e))


# Rerank retrieved chunks using Cross-Encoder relevance scores
def rerank_results(
    query,
    retrieved_chunks,
    top_k=RERANK_TOP_K
):

    try:
        # Validate input
        validate_input(
            query,
            retrieved_chunks,
            top_k
        )

        if not retrieved_chunks:
            logger.warning(
                "No retrieved chunks available for reranking"
            )
            return []

        # Calculate relevance scores
        relevance_scores = calculate_relevance_scores(
            query,
            retrieved_chunks
        )

        # Keep both the cross-encoder and hybrid search signals
        reranked_chunks = []

        for chunk, score in zip(
            retrieved_chunks,
            relevance_scores
        ):
            reranked_chunk = chunk.copy()

            reranked_chunk["rerank_score"] = float(score)

            reranked_chunks.append(reranked_chunk)

        # Rank candidates by cross-encoder score
        cross_encoder_order = sorted(
            reranked_chunks,
            key=lambda chunk: chunk["rerank_score"],
            reverse=True
        )
        cross_encoder_ranks = {
            chunk["id"]: rank
            for rank, chunk in enumerate(cross_encoder_order, start=1)
        }

        # Rank candidates by their original hybrid score
        hybrid_order = sorted(
            reranked_chunks,
            key=lambda chunk: chunk.get("rrf_score", 0.0),
            reverse=True
        )
        hybrid_ranks = {
            chunk["id"]: rank
            for rank, chunk in enumerate(hybrid_order, start=1)
        }

        # Fuse the two rankings so exact sparse matches are not discarded
        for chunk in reranked_chunks:
            cross_rank = cross_encoder_ranks[chunk["id"]]
            hybrid_rank = hybrid_ranks[chunk["id"]]
            chunk["rerank_fused_score"] = (
                CROSS_ENCODER_RANK_WEIGHT
                / (RERANK_RRF_K + cross_rank)
                + HYBRID_RANK_WEIGHT
                / (RERANK_RRF_K + hybrid_rank)
            )

        reranked_chunks.sort(
            key=lambda chunk: chunk["rerank_fused_score"],
            reverse=True
        )

        # Select final top-ranked chunks
        final_results = reranked_chunks[:top_k]

        logger.info(
            f"Reranking completed: "
            f"{len(final_results)} final chunks selected"
        )

        return final_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception(
            "Cross-Encoder reranking failed"
        )
        raise RAGPipelineException(str(e))


# Execute hybrid retrieval followed by Cross-Encoder reranking
def retrieve_and_rerank(
    query,
    top_k=RERANK_TOP_K,
    retrieval_top_k=10,
    user_id=None,
    where=None
):

    try:
        from src.retrieval import hybrid_retrieval

        # Validate query
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string"
            )

        # Validate final result count
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer"
            )

        # Validate candidate retrieval count
        if (
            not isinstance(retrieval_top_k, int)
            or isinstance(retrieval_top_k, bool)
            or retrieval_top_k <= 0
        ):
            raise ValueError(
                "retrieval_top_k must be a positive integer"
            )

        logger.info(
            "Starting hybrid retrieval and reranking pipeline"
        )

        # Step 1: Retrieve candidate chunks using hybrid search
        retrieved_chunks = hybrid_retrieval(
            query=query.strip(),
            top_k=retrieval_top_k,
            user_id=user_id,
            where=where
        )

        # Step 2: Rerank candidates using Cross-Encoder
        final_results = rerank_results(
            query=query.strip(),
            retrieved_chunks=retrieved_chunks,
            top_k=top_k
        )

        logger.info(
            f"Retrieval and reranking completed: "
            f"{len(final_results)} results"
        )

        return final_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception(
            "Retrieval and reranking pipeline failed"
        )
        raise RAGPipelineException(str(e))


# Test Cross-Encoder reranking independently
def main():

    try:
        # Create sample hybrid retrieval results
        sample_results = [
            {
                "id": "chunk_001",
                "text": (
                    "Artificial intelligence enables machines "
                    "to perform tasks requiring human intelligence."
                ),
                "metadata": {
                    "source": "reranking_test.txt",
                    "chunk_id": "chunk_001"
                },
                "rrf_score": 0.025,
                "dense_rank": 2,
                "sparse_rank": 3
            },
            {
                "id": "chunk_002",
                "text": (
                    "Machine learning allows computers to learn "
                    "patterns from data."
                ),
                "metadata": {
                    "source": "reranking_test.txt",
                    "chunk_id": "chunk_002"
                },
                "rrf_score": 0.028,
                "dense_rank": 1,
                "sparse_rank": 1
            },
            {
                "id": "chunk_003",
                "text": (
                    "Deep learning uses neural networks "
                    "to understand complex data."
                ),
                "metadata": {
                    "source": "reranking_test.txt",
                    "chunk_id": "chunk_003"
                },
                "rrf_score": 0.024,
                "dense_rank": 3,
                "sparse_rank": 2
            }
        ]

        query = "How do computers learn from data?"

        print(f"\nSearch query: {query}")

        # Execute Cross-Encoder reranking
        reranked_results = rerank_results(
            query=query,
            retrieved_chunks=sample_results,
            top_k=3
        )

        # Display reranking results
        print("\nCross-Encoder Reranking Results:")

        for rank, result in enumerate(
            reranked_results,
            start=1
        ):
            print(f"\nRank: {rank}")
            print("Chunk ID:", result["id"])
            print("Text:", result["text"])
            print("RRF Score:", result.get("rrf_score"))
            print("Reranking Score:", result["rerank_score"])
            print("Metadata:", result["metadata"])

        print(
            f"\nTotal reranked chunks: {len(reranked_results)}"
        )

    except RAGPipelineException as e:
        logger.error(f"Reranking test failed: {e}")
        print(f"Reranking test failed: {e}")

    except Exception as e:
        logger.exception(
            "Unexpected error in reranking module"
        )
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()


