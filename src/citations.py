
import re

from src.logger import logger
from src.exception import RAGPipelineException


# Extract citation IDs from the generated answer
def extract_citation_ids(answer, retrieved_chunks=None):
    """Extract exact citation IDs, optionally allow-listed by retrieved chunks."""
    try:
        if not isinstance(answer, str):
            raise ValueError("Answer must be a string")
        allowed = None
        if retrieved_chunks is not None:
            if not isinstance(retrieved_chunks, list):
                raise ValueError("Retrieved chunks must be a list")
            allowed = set()
            for chunk in retrieved_chunks:
                if not isinstance(chunk, dict):
                    continue
                if isinstance(chunk.get("id"), str):
                    allowed.add(chunk["id"].strip())
                metadata = chunk.get("metadata", {})
                if isinstance(metadata, dict) and isinstance(metadata.get("chunk_id"), str):
                    allowed.add(metadata["chunk_id"].strip())
        candidates = re.findall(r"\[([^\[\]\r\n]+)\]", answer)
        ids = []
        for value in candidates:
            value = value.strip()
            if not value or re.match(r"(?i)^(source|chunk[\s_-]*id)\s*:", value):
                continue
            if allowed is not None:
                if value not in allowed:
                    continue
            elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
                continue
            ids.append(value)
        ids = list(dict.fromkeys(ids))
        logger.info("Extracted %s valid citation IDs", len(ids))
        return ids
    except Exception as e:
        logger.exception("Citation extraction failed")
        raise RAGPipelineException(str(e)) from e


def verify_citations(citation_ids, retrieved_chunks):
    """Verify IDs against canonical chunk IDs and metadata aliases."""
    try:
        if not isinstance(citation_ids, list) or not isinstance(retrieved_chunks, list):
            raise ValueError("Citation IDs and retrieved chunks must be lists")
        available = set()
        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("id"), str):
                available.add(chunk["id"].strip())
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict) and isinstance(metadata.get("chunk_id"), str):
                available.add(metadata["chunk_id"].strip())
        return [
            {"chunk_id": cid, "verified": isinstance(cid, str) and cid.strip() in available}
            for cid in citation_ids
        ]
    except Exception as e:
        logger.exception("Citation verification failed")
        raise RAGPipelineException(str(e)) from e


def get_citation_sources(verification_results, retrieved_chunks):
    """Resolve verified citation IDs to source metadata."""
    try:
        if not isinstance(verification_results, list) or not isinstance(retrieved_chunks, list):
            raise ValueError("Invalid citation/source inputs")
        lookup = {}
        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue
            ids = [chunk.get("id")]
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict):
                ids.append(metadata.get("chunk_id"))
            for cid in ids:
                if isinstance(cid, str) and cid.strip():
                    lookup[cid.strip()] = chunk
        sources, seen = [], set()
        for item in verification_results:
            if not isinstance(item, dict) or item.get("verified") is not True:
                continue
            cid = item.get("chunk_id")
            chunk = lookup.get(cid)
            if chunk is None or cid in seen:
                continue
            seen.add(cid)
            metadata = chunk.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            sources.append({
                "chunk_id": cid,
                "source": metadata.get("source") or metadata.get("file_name") or metadata.get("filename"),
                "page_number": metadata.get("page_number"),
                "slide_number": metadata.get("slide_number"),
                "verified": True
            })
        return sources
    except Exception as e:
        logger.exception("Failed to retrieve citation sources")
        raise RAGPipelineException(str(e)) from e


def calculate_citation_accuracy(verification_results):

    try:
        if not verification_results:
            return 0.0

        verified_count = sum(
            1 for result in verification_results
            if result.get("verified") is True
        )

        accuracy = verified_count / len(verification_results)

        return round(accuracy, 4)

    except Exception as e:
        logger.exception("Citation accuracy calculation failed")
        raise RAGPipelineException(str(e))


# Calculate an evidence-based confidence indicator
def calculate_confidence_score(
    retrieved_chunks,
    verification_results
):

    try:
        if not isinstance(retrieved_chunks, list):
            raise ValueError(
                "Retrieved chunks must be a list"
            )

        if not isinstance(verification_results, list):
            raise ValueError(
                "Verification results must be a list"
            )

        if not retrieved_chunks:
            return 0.0

        # Calculate the proportion of verified citations
        citation_accuracy = calculate_citation_accuracy(
            verification_results
        )

        # Collect available reranking scores
        rerank_scores = [
            chunk.get("rerank_score")
            for chunk in retrieved_chunks
            if isinstance(chunk.get("rerank_score"), (int, float))
            and not isinstance(chunk.get("rerank_score"), bool)
        ]

        # Normalize reranking scores using sigmoid
        if rerank_scores:

            import math

            rerank_strength = sum(
                1 / (1 + math.exp(-max(-20, min(20, score))))
                for score in rerank_scores
            ) / len(rerank_scores)

        else:
            rerank_strength = 0.0

        # Weighted evidence indicator
        confidence = (
            0.6 * citation_accuracy
            + 0.4 * rerank_strength
        )

        return round(confidence, 4)

    except Exception as e:
        logger.exception("Confidence score calculation failed")
        raise RAGPipelineException(str(e))


# Validate whether the generated answer has sufficient support
def validate_answer_support(
    answer,
    verification_results,
    retrieved_chunks
):

    try:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string")

        if not retrieved_chunks:
            return False

        # Every citation must be verified
        if not verification_results:
            return False

        all_citations_verified = all(
            result.get("verified") is True
            for result in verification_results
        )

        if not all_citations_verified:
            return False

        return True

    except Exception as e:
        logger.exception("Answer support validation failed")
        raise RAGPipelineException(str(e))


# Execute complete citation verification
def process_citations(answer, retrieved_chunks):
    """Run allow-listed citation extraction, verification and reporting."""
    try:
        logger.info("Starting citation verification")
        citation_ids = extract_citation_ids(answer, retrieved_chunks)
        verification = verify_citations(citation_ids, retrieved_chunks)
        sources = get_citation_sources(verification, retrieved_chunks)
        accuracy = calculate_citation_accuracy(verification)
        confidence = calculate_confidence_score(retrieved_chunks, verification)
        supported = validate_answer_support(answer, verification, retrieved_chunks)
        return {
            "answer": answer,
            "citation_ids": citation_ids,
            "citations": sources,
            "citation_verification": verification,
            "citation_accuracy": accuracy,
            "confidence_score": confidence,
            "is_supported": supported
        }
    except RAGPipelineException:
        raise
    except Exception as e:
        logger.exception("Citation processing failed")
        raise RAGPipelineException(str(e)) from e


def main():

    try:
        sample_chunks = [
            {
                "id": "chunk_001",
                "text": (
                    "Machine learning enables systems "
                    "to learn patterns from data."
                ),
                "metadata": {
                    "source": "machine_learning.pdf",
                    "page_number": 2
                },
                "rerank_score": 0.85
            },
            {
                "id": "chunk_002",
                "text": (
                    "Supervised learning uses labelled data."
                ),
                "metadata": {
                    "source": "machine_learning.pdf",
                    "page_number": 3
                },
                "rerank_score": 0.79
            }
        ]

        sample_answer = (
            "Machine learning enables systems to learn "
            "patterns from data. [chunk_001]"
        )

        result = process_citations(
            answer=sample_answer,
            retrieved_chunks=sample_chunks
        )

        print("\nCitation Verification Results:")

        for key, value in result.items():
            print(f"{key}: {value}")

    except RAGPipelineException as e:
        logger.error(f"Citation test failed: {e}")
        print(f"Citation test failed: {e}")

    except Exception as e:
        logger.exception(
            "Unexpected error in citation module"
        )
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
