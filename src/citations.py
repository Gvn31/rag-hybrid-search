import re
import math

from src.logger import logger
from src.exception import RAGPipelineException


# Collect chunk IDs

def get_available_chunk_ids(retrieved_chunks):
    """Collect valid chunk IDs from retrieval results."""

    if not isinstance(retrieved_chunks, list):
        raise ValueError("Retrieved chunks must be a list")

    available_ids = set()

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_id = chunk.get("id")

        if isinstance(chunk_id, str) and chunk_id.strip():
            available_ids.add(chunk_id.strip())

        metadata = chunk.get("metadata", {})

        if isinstance(metadata, dict):
            metadata_id = metadata.get("chunk_id")

            if isinstance(metadata_id, str) and metadata_id.strip():
                available_ids.add(metadata_id.strip())

    return available_ids


# Extract citation IDs

def extract_citation_ids(answer, retrieved_chunks=None):
    """
    Extract citation IDs from generated answers.

    Supports both formats:

        [chunk_id_1] [chunk_id_2]

        [chunk_id_1, chunk_id_2]

    When retrieved_chunks are supplied, only IDs belonging
    to those retrieved chunks are accepted.
    """

    try:
        if not isinstance(answer, str):
            raise ValueError("Answer must be a string")

        allowed_ids = None

        if retrieved_chunks is not None:
            allowed_ids = get_available_chunk_ids(retrieved_chunks)

        # Find all bracketed citation groups.
        candidates = re.findall(
            r"\[([^\[\]\r\n]+)\]",
            answer
        )

        extracted_ids = []

        for candidate in candidates:

            candidate = candidate.strip()

            if not candidate:
                continue

            # Ignore non-citation bracket labels.
            if re.match(
                r"(?i)^(source|chunk[\s_-]*id)\s*:",
                candidate
            ):
                continue

            # IMPORTANT:
            # A model may place multiple IDs inside one bracket.
            # Split comma-separated or semicolon-separated IDs.
            individual_ids = re.split(
                r"[,;]",
                candidate
            )

            for chunk_id in individual_ids:

                chunk_id = chunk_id.strip()

                if not chunk_id:
                    continue

                # Validate against retrieved chunk IDs.
                if allowed_ids is not None:

                    if chunk_id not in allowed_ids:
                        logger.warning(
                            "Ignoring unverified citation ID: {}",
                            chunk_id
                        )
                        continue

                else:

                    # Basic format validation when no allow-list
                    # is provided.
                    if not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}",
                        chunk_id
                    ):
                        continue

                extracted_ids.append(chunk_id)

        # Remove duplicates while preserving original order.
        extracted_ids = list(dict.fromkeys(extracted_ids))

        logger.info(
            "Extracted {} valid citation IDs",
            len(extracted_ids)
        )

        return extracted_ids

    except Exception as e:
        logger.exception("Citation extraction failed")
        raise RAGPipelineException(str(e)) from e


# Verify citations

def verify_citations(citation_ids, retrieved_chunks):
    """Verify extracted IDs against retrieved chunk IDs."""

    try:
        if not isinstance(citation_ids, list):
            raise ValueError("Citation IDs must be a list")

        available_ids = get_available_chunk_ids(retrieved_chunks)

        verification_results = []

        for citation_id in citation_ids:

            verified = (
                isinstance(citation_id, str)
                and citation_id.strip() in available_ids
            )

            verification_results.append({
                "chunk_id": citation_id,
                "verified": verified
            })

        return verification_results

    except Exception as e:
        logger.exception("Citation verification failed")
        raise RAGPipelineException(str(e)) from e


# Resolve citation sources

def get_citation_sources(verification_results, retrieved_chunks):
    """Resolve verified citation IDs to their source metadata."""

    try:
        if not isinstance(verification_results, list):
            raise ValueError("Verification results must be a list")

        if not isinstance(retrieved_chunks, list):
            raise ValueError("Retrieved chunks must be a list")

        lookup = {}

        for chunk in retrieved_chunks:

            if not isinstance(chunk, dict):
                continue

            metadata = chunk.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            chunk_ids = [
                chunk.get("id"),
                metadata.get("chunk_id")
            ]

            for chunk_id in chunk_ids:

                if isinstance(chunk_id, str) and chunk_id.strip():
                    lookup[chunk_id.strip()] = chunk

        sources = []
        seen_ids = set()

        for result in verification_results:

            if not isinstance(result, dict):
                continue

            if result.get("verified") is not True:
                continue

            chunk_id = result.get("chunk_id")

            if not isinstance(chunk_id, str):
                continue

            chunk_id = chunk_id.strip()

            if chunk_id in seen_ids:
                continue

            chunk = lookup.get(chunk_id)

            if chunk is None:
                continue

            metadata = chunk.get("metadata", {})

            if not isinstance(metadata, dict):
                metadata = {}

            sources.append({
                "chunk_id": chunk_id,
                "source": (
                    metadata.get("source")
                    or metadata.get("file_name")
                    or metadata.get("filename")
                ),
                "page_number": metadata.get("page_number"),
                "slide_number": metadata.get("slide_number"),
                "verified": True
            })

            seen_ids.add(chunk_id)

        return sources

    except Exception as e:
        logger.exception("Failed to retrieve citation sources")
        raise RAGPipelineException(str(e)) from e


# Calculate citation accuracy

def calculate_citation_accuracy(verification_results):

    try:
        if not verification_results:
            return 0.0

        verified_count = sum(
            1
            for result in verification_results
            if isinstance(result, dict)
            and result.get("verified") is True
        )

        accuracy = verified_count / len(verification_results)

        return round(accuracy, 4)

    except Exception as e:
        logger.exception("Citation accuracy calculation failed")
        raise RAGPipelineException(str(e)) from e


# Calculate confidence indicator

def calculate_confidence_score(
    retrieved_chunks,
    verification_results
):

    try:
        if not isinstance(retrieved_chunks, list):
            raise ValueError("Retrieved chunks must be a list")

        if not isinstance(verification_results, list):
            raise ValueError("Verification results must be a list")

        if not retrieved_chunks:
            return 0.0

        citation_accuracy = calculate_citation_accuracy(
            verification_results
        )

        rerank_scores = [
            chunk.get("rerank_score")
            for chunk in retrieved_chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get("rerank_score"), (int, float))
            and not isinstance(chunk.get("rerank_score"), bool)
        ]

        if rerank_scores:

            rerank_strength = sum(
                1 / (1 + math.exp(-max(-20, min(20, score))))
                for score in rerank_scores
            ) / len(rerank_scores)

        else:
            rerank_strength = 0.0

        confidence = (
            0.6 * citation_accuracy
            + 0.4 * rerank_strength
        )

        return round(confidence, 4)

    except Exception as e:
        logger.exception("Confidence score calculation failed")
        raise RAGPipelineException(str(e)) from e


# Validate answer support

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

        if not verification_results:
            return False

        all_citations_verified = all(
            isinstance(result, dict)
            and result.get("verified") is True
            for result in verification_results
        )

        return all_citations_verified

    except Exception as e:
        logger.exception("Answer support validation failed")
        raise RAGPipelineException(str(e)) from e


# Process citations

def process_citations(answer, retrieved_chunks):
    """
    Execute citation extraction, verification and reporting.
    """

    try:
        logger.info("Starting citation verification")

        citation_ids = extract_citation_ids(
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )

        verification = verify_citations(
            citation_ids=citation_ids,
            retrieved_chunks=retrieved_chunks
        )

        sources = get_citation_sources(
            verification_results=verification,
            retrieved_chunks=retrieved_chunks
        )

        accuracy = calculate_citation_accuracy(
            verification
        )

        confidence = calculate_confidence_score(
            retrieved_chunks=retrieved_chunks,
            verification_results=verification
        )

        supported = validate_answer_support(
            answer=answer,
            verification_results=verification,
            retrieved_chunks=retrieved_chunks
        )

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


# Run citation examples

def main():

    try:
        sample_chunks = [
            {
                "id": "chunk_001",
                "text": "Machine learning enables systems to learn patterns.",
                "metadata": {
                    "chunk_id": "chunk_001",
                    "source": "machine_learning.pdf",
                    "page_number": 2
                },
                "rerank_score": 0.85
            },
            {
                "id": "chunk_002",
                "text": "Supervised learning uses labelled data.",
                "metadata": {
                    "chunk_id": "chunk_002",
                    "source": "machine_learning.pdf",
                    "page_number": 3
                },
                "rerank_score": 0.79
            },
            {
                "id": "chunk_003",
                "text": "Unsupervised learning discovers patterns.",
                "metadata": {
                    "chunk_id": "chunk_003",
                    "source": "machine_learning.pdf",
                    "page_number": 4
                },
                "rerank_score": 0.75
            }
        ]

        test_answers = [
            (
                "Single citation: Machine learning learns from data "
                "[chunk_001]."
            ),
            (
                "Separate citations: Machine learning learns from data "
                "[chunk_001] [chunk_002]."
            ),
            (
                "Combined citations: Machine learning learns from data "
                "[chunk_001, chunk_002]."
            ),
            (
                "Three combined citations: Machine learning learns from "
                "data [chunk_001, chunk_002, chunk_003]."
            ),
            (
                "Mixed citations: Machine learning learns from data "
                "[chunk_001] and supervised learning uses labelled data "
                "[chunk_002, chunk_003]."
            ),
            (
                "Invalid citation: Machine learning learns from data "
                "[fake_chunk_999]."
            )
        ]

        print("\nCITATION EXTRACTION TESTS")
        print("=" * 65)

        for index, answer in enumerate(test_answers, start=1):

            result = process_citations(
                answer=answer,
                retrieved_chunks=sample_chunks
            )

            print(f"\nTEST {index}")
            print("-" * 65)

            print("Answer:")
            print(answer)

            print("\nExtracted IDs:")
            print(result["citation_ids"])

            print("\nCitation accuracy:")
            print(result["citation_accuracy"])

            print("\nVerified citations:")
            print(result["citation_verification"])

            print("\nAnswer supported:")
            print(result["is_supported"])

        print("\nAll citation extraction tests completed.")

    except RAGPipelineException as e:
        logger.error("Citation test failed: {}", e)
        print(f"Citation test failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in citation module")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
