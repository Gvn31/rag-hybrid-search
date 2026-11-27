import math
import re
from pathlib import Path

from src.config import (
    TOP_K,
    DENSE_WEIGHT,
    SPARSE_WEIGHT
)

from src.logger import logger
from src.exception import RAGPipelineException

from src.embeddings import get_embedding_model
from src.vector_store import search_similar_chunks
from src.bm25_store import search_bm25


# Weighted Reciprocal Rank Fusion configuration
RRF_K = 60
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def infer_explicit_source_filter(query):
    """Limit retrieval when the question explicitly names a source file/type."""
    if not isinstance(query, str) or not query.strip():
        return None

    try:
        raw_files = [
            path for path in RAW_DATA_DIR.iterdir()
            if path.is_file()
        ] if RAW_DATA_DIR.is_dir() else []
        lowered_query = query.casefold()

        for path in raw_files:
            if path.name.casefold() in lowered_query:
                return {"source": path.name}

        source_types = {
            ".pdf": r"(?:pdf|portable document format)",
            ".pptx": r"(?:pptx|powerpoint|slide deck|presentation|presentations|slides?)",
            ".docx": r"(?:docx|word document)",
            ".html": r"(?:html|web page)",
            ".md": r"(?:markdown|md file)",
            ".txt": r"(?:txt file|plain text file)"
        }
        source_context = re.compile(
            r"\b(?:according to|in|from|within|described in|listed in)\s+"
            r"(?:the\s+)?(?P<type>pdf|portable document format|pptx|"
            r"powerpoint|slide deck|presentations?|slides?|docx|word document|html|web page|"
            r"markdown|md file|txt file|plain text file)\b"
            r"|\bthe\s+(?P<type2>pdf|pptx|powerpoint|presentations?|slides?|docx|html|"
            r"markdown)\s+(?:says?|states?|lists?|shows?|describes?)\b",
            flags=re.IGNORECASE
        )
        match = source_context.search(query)
        if not match:
            return None

        named_type = (match.group("type") or match.group("type2") or "").casefold()
        extension = next(
            (
                suffix for suffix, pattern in source_types.items()
                if re.fullmatch(pattern, named_type, flags=re.IGNORECASE)
            ),
            None
        )
        if extension is None:
            return None

        matching_sources = sorted(
            path.name for path in raw_files
            if path.suffix.casefold() == extension
        )
        if not matching_sources:
            return None
        if len(matching_sources) == 1:
            return {"source": matching_sources[0]}
        return {"source": {"$in": matching_sources}}

    except Exception as e:
        logger.warning("Could not infer a source filter from the query: {}", e)
        return None


def combine_where_filters(where, inferred_filter):
    if inferred_filter is None:
        return where
    if where is None:
        return inferred_filter
    return {"$and": [where, inferred_filter]}


# Validate retrieval parameters
def validate_retrieval_parameters(top_k):
    try:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer"
            )

        return top_k

    except Exception as e:
        raise RAGPipelineException(str(e)) from e


# Validate RRF weights
def validate_rrf_weights(dense_weight, sparse_weight):
    try:
        weights = [dense_weight, sparse_weight]

        for weight in weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError("RRF weights must be numeric")

            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    "RRF weights must be finite and non-negative"
                )

        if dense_weight + sparse_weight <= 0:
            raise ValueError(
                "At least one RRF weight must be positive"
            )

        return float(dense_weight), float(sparse_weight)

    except Exception as e:
        raise RAGPipelineException(str(e)) from e


# Generate an embedding for the user's search query
def generate_query_embedding(query):
    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Load the same embedding model used for document chunks
        embedding_model = get_embedding_model()

        # Generate query embedding directly
        query_embedding = embedding_model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError(
                "Query embedding generation returned an invalid shape"
            )

        embedding = query_embedding[0].tolist()

        if not embedding:
            raise ValueError(
                "Query embedding generation returned an empty vector"
            )

        if not all(math.isfinite(float(value)) for value in embedding):
            raise ValueError(
                "Query embedding contains invalid numeric values"
            )

        logger.info("Query embedding generated successfully")

        return embedding

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Query embedding generation failed")
        raise RAGPipelineException(str(e)) from e


# Perform dense retrieval using ChromaDB
def dense_retrieval(
    query_embedding,
    top_k=TOP_K,
    user_id=None,
    where=None
):
    try:
        validate_retrieval_parameters(top_k)

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
        raise RAGPipelineException(str(e)) from e


# Perform sparse retrieval using BM25
def sparse_retrieval(
    query,
    top_k=TOP_K,
    user_id=None,
    where=None
):
    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")

        validate_retrieval_parameters(top_k)

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
        raise RAGPipelineException(str(e)) from e


# Validate retrieval result records before fusion
def validate_retrieval_results(results, retrieval_type):
    try:
        if not isinstance(results, list):
            raise ValueError(
                f"{retrieval_type} results must be a list"
            )

        seen_ids = set()

        for result in results:

            if not isinstance(result, dict):
                raise ValueError(
                    f"Each {retrieval_type} result must be a dictionary"
                )

            if not all(
                key in result
                for key in ["id", "text", "metadata"]
            ):
                raise ValueError(
                    f"{retrieval_type} result must contain id, text and metadata"
                )

            chunk_id = result["id"]

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"{retrieval_type} result contains an invalid ID"
                )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate ID in {retrieval_type} results: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            if not isinstance(result["text"], str):
                raise ValueError(
                    f"{retrieval_type} result text must be a string"
                )

            if not isinstance(result["metadata"], dict):
                raise ValueError(
                    f"{retrieval_type} result metadata must be a dictionary"
                )

        return True

    except Exception as e:
        raise RAGPipelineException(str(e)) from e


# Combine dense and sparse rankings using weighted RRF
def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    top_k=TOP_K,
    dense_weight=DENSE_WEIGHT,
    sparse_weight=SPARSE_WEIGHT
):
    try:
        validate_retrieval_parameters(top_k)

        validate_rrf_weights(
            dense_weight,
            sparse_weight
        )

        validate_retrieval_results(
            dense_results,
            "Dense"
        )

        validate_retrieval_results(
            sparse_results,
            "Sparse"
        )

        # Store fused scores and chunk information
        fused_results = {}

        # Process dense retrieval rankings
        for rank, result in enumerate(dense_results, start=1):

            chunk_id = result["id"]

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

            chunk_id = result["id"]

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

        # Sort chunks by combined RRF score
        ranked_results = sorted(
            fused_results.values(),
            key=lambda result: result["rrf_score"],
            reverse=True
        )

        final_results = ranked_results[:top_k]

        logger.info(
            f"RRF generated {len(final_results)} final results"
        )

        return final_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Reciprocal Rank Fusion failed")
        raise RAGPipelineException(str(e)) from e


# Add a focused secondary query for scanned-page chunk checks.
def build_retrieval_queries(query):
    base_query = query.strip()
    queries = [base_query]
    lowered = base_query.casefold()
    asks_about_ocr = bool(
        re.search(r"\b(?:scanned|scan|ocr)\b", lowered)
    )
    asks_to_check_chunks = bool(
        re.search(r"\b(?:check|verify|validate|inspect)\w*\b", lowered)
        and re.search(r"\bchunks?\b", lowered)
    )

    if asks_about_ocr and asks_to_check_chunks:
        queries.append(
            "scanned-page OCR chunk quality checks missing content "
            "OCR errors chunk-size validation"
        )

    return list(dict.fromkeys(queries))


# Merge rankings from the original and expanded queries.
def merge_query_results(result_lists, query_weights, top_k):
    validate_retrieval_parameters(top_k)
    if len(result_lists) != len(query_weights):
        raise ValueError("Each query result list must have a query weight")

    fused_results = {}
    for results, weight in zip(result_lists, query_weights):
        seen_ids = set()
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = dict(result)
                fused_results[chunk_id]["rrf_score"] = 0.0

            fused_results[chunk_id]["rrf_score"] += (
                weight / (RRF_K + rank)
            )

    ranked_results = sorted(
        fused_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True
    )
    return ranked_results[:top_k]


# Execute the complete hybrid retrieval pipeline.
def hybrid_retrieval(
    query,
    top_k=TOP_K,
    user_id=None,
    where=None,
    dense_weight=DENSE_WEIGHT,
    sparse_weight=SPARSE_WEIGHT
):
    try:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty")

        validate_retrieval_parameters(top_k)
        validate_rrf_weights(dense_weight, sparse_weight)

        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string")

        logger.info("Starting hybrid retrieval")
        inferred_filter = infer_explicit_source_filter(query)
        effective_where = combine_where_filters(where, inferred_filter)
        if inferred_filter:
            logger.info("Applying explicit source filter: {}", inferred_filter)

        query_variants = build_retrieval_queries(query)
        query_results = []

        for query_index, search_query in enumerate(query_variants):
            query_embedding = generate_query_embedding(search_query)
            dense_results = dense_retrieval(
                query_embedding=query_embedding,
                top_k=top_k,
                user_id=user_id,
                where=effective_where
            )
            sparse_results = sparse_retrieval(
                query=search_query,
                top_k=top_k,
                user_id=user_id,
                where=effective_where
            )
            fused_results = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
            query_results.append(fused_results)

            logger.info(
                "Hybrid query variant {}/{} returned {} candidates",
                query_index + 1,
                len(query_variants),
                len(fused_results)
            )

        if len(query_results) == 1:
            final_results = query_results[0]
        else:
            final_results = merge_query_results(
                result_lists=query_results,
                query_weights=[0.65, 0.35],
                top_k=top_k
            )

        logger.info(
            "Hybrid retrieval completed with {} results",
            len(final_results)
        )
        return final_results

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Hybrid retrieval failed")
        raise RAGPipelineException(str(e)) from e


# Test the complete hybrid retrieval pipeline
def main():
    try:
        from src.embeddings import generate_embeddings

        from src.vector_store import (
            store_embeddings,
            delete_document as delete_vector_document
        )

        from src.bm25_store import (
            index_chunks,
            delete_document as delete_bm25_document
        )

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
                "Deep learning uses neural networks "
                "to understand complex data."
            )
        ]

        sample_chunks = []

        current_position = 0

        # Create sample chunks with complete metadata
        for index, text in enumerate(sample_texts):

            start_index = current_position
            end_index = start_index + len(text)

            sample_chunks.append({
                "text": text,
                "metadata": {
                    "source": "hybrid_test.txt",
                    "chunk_id": f"hybrid_test_{index + 1:03d}",
                    "chunk_index": index,
                    "chunking_strategy": "test",
                    "char_count": len(text),
                    "start_index": start_index,
                    "end_index": end_index,
                    "section_heading": None
                }
            })

            current_position = end_index + 1

        # Generate embeddings for the sample document
        embedded_chunks = generate_embeddings(sample_chunks)

        # Store document chunks in both indexes
        vector_count = store_embeddings(embedded_chunks)
        bm25_count = index_chunks(sample_chunks)

        print(f"\nChromaDB chunks stored: {vector_count}")
        print(f"BM25 chunks indexed: {bm25_count}")

        # Define the test search query
        query = "How do computers learn from data?"

        print(f"\nHybrid search query: {query}")

        # Execute hybrid retrieval
        results = hybrid_retrieval(
            query=query,
            top_k=3
        )

        # Display fused results
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

        # Remove only the test document from both indexes
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
        logger.exception(
            "Unexpected error in retrieval module"
        )
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()






