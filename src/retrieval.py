
from src.config import (
    TOP_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT
)

from src.logger import logger
from src.exception import RAGPipelineException

from src.embeddings import generate_embeddings
from src.vector_store import search_similar_chunks
from src.bm25_store import search_bm25


# Define the RRF ranking constant
RRF_K = 60


# Generate an embedding for the user's search query
def generate_query_embedding(query):
    try:
        # Validate the search query
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Prepare the query in the same format as document chunks
        query_data = [
            {
                "text": query.strip(),
                "metadata": {
                    "source": "user_query"
                }
            }
        ]

        # Generate the query vector using the existing embedding model
        embedded_query = generate_embeddings(query_data)

        if not embedded_query:
            raise ValueError("Query embedding generation returned empty")

        query_embedding = embedded_query[0]["embedding"]

        logger.info("Query embedding generated successfully")

        return query_embedding

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Query embedding generation failed")
        raise RAGPipelineException(str(e))


# Perform dense retrieval using ChromaDB
def dense_retrieval(
    query_embedding,
    top_k=TOP_K,
    user_id=None,
    where=None
):
    try:
        # Search for semantically similar document chunks
        dense_results = search_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            user_id=user_id,
            where=where
        )

        logger.info(
            f"Dense retrieval returned {len(dense_results)} results"
        )

        return dense_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Dense retrieval failed")
        raise RAGPipelineException(str(e))


# Perform sparse retrieval using BM25
def sparse_retrieval(
    query,
    top_k=TOP_K,
    user_id=None,
    where=None
):
    try:
        # Search for keyword-relevant document chunks
        sparse_results = search_bm25(
            query=query,
            top_k=top_k,
            user_id=user_id,
            where=where
        )

        logger.info(
            f"Sparse retrieval returned {len(sparse_results)} results"
        )

        return sparse_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Sparse retrieval failed")
        raise RAGPipelineException(str(e))


# Combine dense and sparse rankings using weighted RRF
def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    top_k=TOP_K,
    dense_weight=DENSE_WEIGHT,
    sparse_weight=SPARSE_WEIGHT
):
    try:
        # Validate the retrieval results
        if not isinstance(dense_results, list):
            raise ValueError("Dense results must be a list")

        if not isinstance(sparse_results, list):
            raise ValueError("Sparse results must be a list")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("RRF weights cannot be negative")

        if dense_weight + sparse_weight == 0:
            raise ValueError("At least one RRF weight must be positive")

        # Store fused scores and chunk information
        fused_results = {}

        # Process dense retrieval rankings
        for rank, result in enumerate(dense_results, start=1):

            chunk_id = result.get("id")

            if not chunk_id:
                raise ValueError("Dense result is missing its ID")

            # Calculate the weighted dense RRF score
            rrf_score = dense_weight / (RRF_K + rank)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_distance": None,
                    "sparse_score": None
                }

            fused_results[chunk_id]["rrf_score"] += rrf_score
            fused_results[chunk_id]["dense_rank"] = rank

            fused_results[chunk_id]["dense_distance"] = (
                result.get("distance")
            )

        # Process sparse retrieval rankings
        for rank, result in enumerate(sparse_results, start=1):

            chunk_id = result.get("id")

            if not chunk_id:
                raise ValueError("Sparse result is missing its ID")

            # Calculate the weighted sparse RRF score
            rrf_score = sparse_weight / (RRF_K + rank)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    "id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "dense_distance": None,
                    "sparse_score": None
                }

            fused_results[chunk_id]["rrf_score"] += rrf_score
            fused_results[chunk_id]["sparse_rank"] = rank

            fused_results[chunk_id]["sparse_score"] = (
                result.get("score")
            )

        # Sort all chunks by their combined RRF score
        ranked_results = sorted(
            fused_results.values(),
            key=lambda result: result["rrf_score"],
            reverse=True
        )

        # Return only the requested number of results
        final_results = ranked_results[:top_k]

        logger.info(
            f"RRF generated {len(final_results)} final results"
        )

        return final_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Reciprocal Rank Fusion failed")
        raise RAGPipelineException(str(e))


# Execute the complete hybrid retrieval pipeline
def hybrid_retrieval(
    query,
    top_k=TOP_K,
    user_id=None,
    where=None,
    dense_weight=DENSE_WEIGHT,
    sparse_weight=SPARSE_WEIGHT
):
    try:
        # Validate the query
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")

        logger.info("Starting hybrid retrieval")

        # Step 1: Generate the query embedding
        query_embedding = generate_query_embedding(query)

        # Step 2: Perform dense vector retrieval
        dense_results = dense_retrieval(
            query_embedding=query_embedding,
            top_k=top_k,
            user_id=user_id,
            where=where
        )

        # Step 3: Perform sparse keyword retrieval
        sparse_results = sparse_retrieval(
            query=query,
            top_k=top_k,
            user_id=user_id,
            where=where
        )

        # Step 4: Combine both retrieval rankings using weighted RRF
        final_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )

        logger.info(
            f"Hybrid retrieval completed with "
            f"{len(final_results)} results"
        )

        return final_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Hybrid retrieval failed")
        raise RAGPipelineException(str(e))


# Test the complete hybrid retrieval pipeline
def main():
    try:
        from src.vector_store import (
            store_embeddings,
            delete_document as delete_vector_document
        )

        from src.bm25_store import (
            index_chunks,
            delete_document as delete_bm25_document
        )

        # Create sample document chunks for testing
        sample_chunks = [
            {
                "text": (
                    "Artificial intelligence enables machines "
                    "to perform tasks requiring human intelligence."
                ),
                "metadata": {
                    "source": "hybrid_test.txt",
                    "chunk_id": "hybrid_test_001",
                    "chunk_index": 0
                }
            },
            {
                "text": (
                    "Machine learning allows computers to learn "
                    "patterns from data."
                ),
                "metadata": {
                    "source": "hybrid_test.txt",
                    "chunk_id": "hybrid_test_002",
                    "chunk_index": 1
                }
            },
            {
                "text": (
                    "Deep learning uses neural networks "
                    "to understand complex data."
                ),
                "metadata": {
                    "source": "hybrid_test.txt",
                    "chunk_id": "hybrid_test_003",
                    "chunk_index": 2
                }
            }
        ]

        # Generate embeddings for the sample chunks
        embedded_chunks = generate_embeddings(sample_chunks)

        # Store chunks in the dense vector database
        vector_count = store_embeddings(embedded_chunks)

        # Store chunks in the sparse BM25 index
        bm25_count = index_chunks(sample_chunks)

        print(f"\nChromaDB chunks stored: {vector_count}")
        print(f"BM25 chunks indexed: {bm25_count}")

        # Define the test search query
        query = "How do computers learn from data?"

        print(f"\nHybrid search query: {query}")

        # Execute the complete hybrid retrieval pipeline
        results = hybrid_retrieval(
            query=query,
            top_k=3
        )

        # Display the final fused results
        print("\nHybrid Retrieval Results:")

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print("Chunk ID:", result["id"])
            print("Text:", result["text"])
            print("RRF Score:", result["rrf_score"])
            print("Dense Rank:", result["dense_rank"])
            print("Sparse Rank:", result["sparse_rank"])
            print("Dense Distance:", result["dense_distance"])
            print("Sparse Score:", result["sparse_score"])
            print("Metadata:", result["metadata"])

        # Remove only the sample test document from both indexes
        deleted_vector_count = delete_vector_document(
            source="hybrid_test.txt"
        )

        deleted_bm25_count = delete_bm25_document(
            source="hybrid_test.txt"
        )

        print(
            f"\nChromaDB test chunks deleted: {deleted_vector_count}"
        )

        print(
            f"BM25 test chunks deleted: {deleted_bm25_count}"
        )

    except RAGPipelineException as e:
        logger.error(f"Hybrid retrieval test failed: {e}")
        print(f"Hybrid retrieval test failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in retrieval module")
        print(f"Unexpected error: {e}")


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()
