import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone

from src.config import EVALUATION_DIR
from src.logger import logger
from src.exception import RAGPipelineException


QA_DATASET_PATH = EVALUATION_DIR / "qa_dataset.json"
EVALUATION_REPORT_PATH = EVALUATION_DIR / "evaluation_report.json"

MIN_QA_QUESTIONS = 50

VALID_QUESTION_TYPES = {
    "direct",
    "multi_hop",
    "ambiguous",
    "unanswerable"
}

EVALUATION_SOURCES = [
    "api_documentation_sample.html",
    "compliance_policy_sample.docx",
    "knowledge_base_sample.md",
    "retrieval-augmented-generation-options.pdf",
    "sample-document-medium.docx",
    "support_transcripts_sample.txt"
]


def normalize_qa_dataset(dataset):
    """Convert the JSON dataset into the evaluator's format."""

    try:
        if isinstance(dataset, dict):
            questions = dataset.get("questions")

        elif isinstance(dataset, list):
            questions = dataset

        else:
            raise ValueError("Invalid QA dataset format")

        if not isinstance(questions, list):
            raise ValueError("Dataset must contain a questions list")

        normalized_questions = []

        for item in questions:
            if not isinstance(item, dict):
                raise ValueError("Invalid QA record")

            question_id = item.get(
                "id",
                item.get("question_id")
            )

            category = item.get(
                "category",
                item.get("question_type", "direct")
            )

            question_type = (
                "direct"
                if category == "straightforward"
                else category
            )

            answerable = item.get(
                "answerable",
                question_type != "unanswerable"
            )

            if not isinstance(answerable, bool):
                raise ValueError(
                    f"Invalid answerable value: {question_id}"
                )

            source_chunks = item.get("source_chunks", [])

            if not isinstance(source_chunks, list):
                raise ValueError(
                    f"Invalid source_chunks: {question_id}"
                )

            relevant_chunk_ids = item.get(
                "relevant_chunk_ids",
                []
            )

            if not relevant_chunk_ids and answerable:
                relevant_chunk_ids = [
                    chunk["chunk_id"]
                    for chunk in source_chunks
                    if (
                        isinstance(chunk, dict)
                        and isinstance(
                            chunk.get("chunk_id"),
                            str
                        )
                    )
                ]

            if not answerable:
                relevant_chunk_ids = []

            normalized_questions.append({
                "question_id": question_id,
                "question": item.get("question"),
                "question_type": question_type,
                "expected_answer": item.get(
                    "expected_answer",
                    ""
                ),
                "relevant_chunk_ids": relevant_chunk_ids,
                "answerable": answerable
            })

        return normalized_questions

    except Exception as e:
        logger.exception("QA dataset normalization failed")
        raise RAGPipelineException(str(e)) from e


def normalize_text(text):
    """Normalize text for lexical scoring; this is not semantic evaluation."""
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        text = text.casefold()
        text = re.sub(r"\[[^\[\]]+\]", " ", text)
        contractions = {
            r"\bcan't\b": "cannot", r"\bwon't\b": "will not",
            r"\bdon't\b": "do not", r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not", r"\bisn't\b": "is not",
            r"\baren't\b": "are not", r"\bwasn't\b": "was not",
            r"\bweren't\b": "were not", r"\bi've\b": "i have",
            r"\bi'm\b": "i am", r"\bit's\b": "it is",
            r"\bthey're\b": "they are", r"\bwe're\b": "we are",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        number_words = {
            "zero":"0", "one":"1", "two":"2", "three":"3", "four":"4",
            "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9",
            "ten":"10", "eleven":"11", "twelve":"12", "thirteen":"13",
            "fourteen":"14", "fifteen":"15", "sixteen":"16",
            "seventeen":"17", "eighteen":"18", "nineteen":"19",
            "twenty":"20",
        }
        tokens = [number_words.get(token, token) for token in text.split()]
        return re.sub(r"\s+", " ", " ".join(tokens)).strip()
    except Exception as e:
        logger.exception("Text normalization failed")
        raise RAGPipelineException(str(e)) from e


def calculate_token_metrics(predicted, expected):
    """Calculate token precision, recall and F1."""

    try:
        predicted_tokens = normalize_text(predicted).split()
        expected_tokens = normalize_text(expected).split()

        if not predicted_tokens or not expected_tokens:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        predicted_counts = {}
        expected_counts = {}

        for token in predicted_tokens:
            predicted_counts[token] = (
                predicted_counts.get(token, 0) + 1
            )

        for token in expected_tokens:
            expected_counts[token] = (
                expected_counts.get(token, 0) + 1
            )

        common_tokens = sum(
            min(count, expected_counts.get(token, 0))
            for token, count in predicted_counts.items()
        )

        precision = common_tokens / len(predicted_tokens)
        recall = common_tokens / len(expected_tokens)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    except Exception as e:
        logger.exception("Token metric calculation failed")
        raise RAGPipelineException(str(e)) from e


def evaluate_answer_correctness(predicted_answer, expected_answer):
    """Compare the generated answer with the expected answer."""

    try:
        predicted_normalized = normalize_text(predicted_answer)
        expected_normalized = normalize_text(expected_answer)

        exact_match = (
            predicted_normalized == expected_normalized
        )

        token_metrics = calculate_token_metrics(
            predicted_answer,
            expected_answer
        )

        return {
            "exact_match": int(exact_match),
            "token_precision": token_metrics["precision"],
            "token_recall": token_metrics["recall"],
            "token_f1": token_metrics["f1"]
        }

    except Exception as e:
        logger.exception("Answer correctness evaluation failed")
        raise RAGPipelineException(str(e)) from e


def get_retrieved_chunk_ids(retrieved_chunks):
    """Get both retrieval IDs and original chunk IDs."""

    try:
        retrieved_ids = []

        for chunk in retrieved_chunks:
            if not isinstance(chunk, dict):
                continue

            chunk_id = chunk.get("id")

            if isinstance(chunk_id, str):
                retrieved_ids.append(chunk_id)

            metadata = chunk.get("metadata", {})

            if isinstance(metadata, dict):
                metadata_chunk_id = metadata.get("chunk_id")

                if isinstance(metadata_chunk_id, str):
                    retrieved_ids.append(metadata_chunk_id)

        return list(dict.fromkeys(retrieved_ids))

    except Exception as e:
        logger.exception("Retrieved chunk ID extraction failed")
        raise RAGPipelineException(str(e)) from e


def calculate_recall_at_k(retrieved_chunks, relevant_chunk_ids):
    """Calculate Recall@K."""

    try:
        if not relevant_chunk_ids:
            return None

        retrieved_ids = set(
            get_retrieved_chunk_ids(retrieved_chunks)
        )

        relevant_ids = set(relevant_chunk_ids)

        matched = retrieved_ids.intersection(relevant_ids)

        return round(
            len(matched) / len(relevant_ids),
            4
        )

    except Exception as e:
        logger.exception("Recall@K calculation failed")
        raise RAGPipelineException(str(e)) from e


def calculate_mrr(retrieved_chunks, relevant_chunk_ids):
    """Calculate reciprocal rank using one rank per retrieved chunk."""
    try:
        if not relevant_chunk_ids:
            return None
        relevant_ids = set(relevant_chunk_ids)
        # A chunk's id and metadata.chunk_id are aliases, not separate ranks.
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if not isinstance(chunk, dict):
                continue
            ids = set()
            if isinstance(chunk.get("id"), str):
                ids.add(chunk["id"])
            metadata = chunk.get("metadata", {})
            if isinstance(metadata, dict) and isinstance(metadata.get("chunk_id"), str):
                ids.add(metadata["chunk_id"])
            if ids.intersection(relevant_ids):
                return round(1.0 / rank, 4)
        return 0.0
    except Exception as e:
        logger.exception("MRR calculation failed")
        raise RAGPipelineException(str(e)) from e


def evaluate_citation_accuracy(citation_results):
    """Calculate the percentage of verified citations."""

    try:
        if not isinstance(citation_results, list):
            raise ValueError("Citation results must be a list")

        if not citation_results:
            return None

        verified_count = sum(
            1
            for result in citation_results
            if (
                isinstance(result, dict)
                and result.get("verified") is True
            )
        )

        return round(
            verified_count / len(citation_results),
            4
        )

    except Exception as e:
        logger.exception("Citation accuracy evaluation failed")
        raise RAGPipelineException(str(e)) from e


def evaluate_lexical_faithfulness(answer, retrieved_chunks):
    """Calculate a lexical overlap proxy for faithfulness."""

    try:
        answer_tokens = normalize_text(answer).split()

        context = " ".join(
            chunk.get("text", "")
            for chunk in retrieved_chunks
            if (
                isinstance(chunk, dict)
                and isinstance(chunk.get("text"), str)
            )
        )

        context_tokens = set(
            normalize_text(context).split()
        )

        if not answer_tokens or not context_tokens:
            return 0.0

        supported_tokens = sum(
            1
            for token in answer_tokens
            if token in context_tokens
        )

        return round(
            supported_tokens / len(answer_tokens),
            4
        )

    except Exception as e:
        logger.exception("Lexical faithfulness evaluation failed")
        raise RAGPipelineException(str(e)) from e




def is_abstention(answer):
    """Detect explicit inability to answer; qualified answers remain answers."""
    try:
        if not isinstance(answer, str) or not answer.strip():
            return False
        normalized = normalize_text(answer)
        custom_pricing = (
            re.search(r"\b(custom priced|custom pricing|price is custom)\b", normalized)
            and re.search(r"\b(plan|enterprise|subscription|price|pricing)\b", normalized)
        )
        if custom_pricing:
            return False
        patterns = [
            r"\bi do not know\b", r"\bi cannot answer\b",
            r"\bcannot answer (this|the question)\b", r"\bunable to answer\b",
            r"\bi am unable to (determine|identify|provide|find)\b",
            r"\b(i cannot|cannot) (determine|identify|establish|verify)\b",
            r"\b(do not|does not) (specify|state|identify|mention|provide|indicate)\b",
            r"\b(not specified|not identified|not stated|not mentioned)\b",
            r"\b(not provided|not available) (in|by) (the )?(provided )?(documents|document|context|source)\b",
            r"\bnot contained in (the )?(provided )?(context|documents)\b",
            r"\bdoes not contain (the requested|any relevant|enough) information\b",
            r"\bdoes not provide (the requested|any|sufficient) information\b",
            r"\bno information (is )?(available|provided)\b",
            r"\binsufficient information\b", r"\bnot enough information\b",
            r"\bcannot be determined from (the )?(provided )?(documents|document|context)\b",
            r"\bthe (provided )?(documents|document|context) do not (say|state|specify|identify|provide)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in patterns)
    except Exception as e:
        logger.exception("Abstention detection failed")
        raise RAGPipelineException(str(e)) from e


def is_clarification(answer):
    """Detect a request for missing scope; a question mark alone is insufficient."""
    try:
        if not isinstance(answer, str) or not answer.strip():
            return False
        text = normalize_text(answer)
        patterns = [
            r"\b(could you|can you|please) clarify\b",
            r"\b(could you|can you|please) specify\b",
            r"\bwhich (specific )?(plan|tier|database|type|kind|retention period|encryption method|security area|aspect|service|option)\b",
            r"\bwhat (specific )?(plan|tier|database|type|kind|retention period|encryption method|security area|aspect|service|option)\b",
            r"\b(which|what) .* (are|do) you referring to\b",
            r"\b(are you asking about|do you mean|are you referring to)\b",
            r"\b(please )?(share|provide) (your )?(requirements|use case|workload|specific details|more details)\b",
            r"\bwhat are your (main )?(requirements|needs|priorities|use case)\b",
            r"\bwhich .* would you like\b",
            r"\bwhat specifically would you like\b",
            r"\bdo you mean .* or\b",
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    except Exception as e:
        logger.exception("Clarification detection failed")
        raise RAGPipelineException(str(e)) from e


def validate_qa_dataset(dataset, min_questions=MIN_QA_QUESTIONS):
    """Validate the dataset after converting it to the internal format."""

    try:
        normalized_dataset = normalize_qa_dataset(dataset)

        if len(normalized_dataset) < min_questions:
            raise ValueError(
                f"QA dataset must contain at least {min_questions} questions"
            )

        question_ids = set()

        for index, item in enumerate(normalized_dataset):
            question_id = item.get("question_id")
            question = item.get("question")
            question_type = item.get("question_type")
            expected_answer = item.get("expected_answer")
            relevant_chunk_ids = item.get("relevant_chunk_ids")

            if (
                not isinstance(question_id, str)
                or not question_id.strip()
            ):
                raise ValueError(
                    f"Invalid question ID at index {index}"
                )

            if question_id in question_ids:
                raise ValueError(
                    f"Duplicate question ID: {question_id}"
                )

            question_ids.add(question_id)

            if (
                not isinstance(question, str)
                or not question.strip()
            ):
                raise ValueError(
                    f"Invalid question at index {index}"
                )

            if question_type not in VALID_QUESTION_TYPES:
                raise ValueError(
                    f"Invalid question type: {question_type}"
                )

            if (
                not isinstance(expected_answer, str)
                or not expected_answer.strip()
            ):
                raise ValueError(
                    f"Invalid expected answer: {question_id}"
                )

            if not isinstance(relevant_chunk_ids, list):
                raise ValueError(
                    f"Invalid relevant chunk IDs: {question_id}"
                )

            if (
                question_type != "unanswerable"
                and not relevant_chunk_ids
            ):
                raise ValueError(
                    f"Missing source chunk IDs: {question_id}"
                )

            if not all(
                isinstance(chunk_id, str) and chunk_id.strip()
                for chunk_id in relevant_chunk_ids
            ):
                raise ValueError(
                    f"Invalid chunk ID: {question_id}"
                )

        logger.info(
            f"Validated QA dataset with {len(normalized_dataset)} questions"
        )

        return True

    except Exception as e:
        logger.exception("QA dataset validation failed")
        raise RAGPipelineException(str(e)) from e


def load_qa_dataset(file_path=QA_DATASET_PATH):
    """Load and validate the QA dataset."""

    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"QA dataset not found: {file_path}"
            )

        with file_path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)

        validate_qa_dataset(dataset)

        return normalize_qa_dataset(dataset)

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Failed to load QA dataset")
        raise RAGPipelineException(str(e)) from e


def build_evaluation_filter(source_filter=None):
    """Build the source filter for evaluation documents."""

    try:
        sources = (
            source_filter
            if source_filter is not None
            else EVALUATION_SOURCES
        )

        if not isinstance(sources, list) or not sources:
            raise ValueError(
                "Evaluation source filter must be a non-empty list"
            )

        if not all(
            isinstance(source, str) and source.strip()
            for source in sources
        ):
            raise ValueError("Invalid evaluation source")

        if len(sources) == 1:
            return {"source": sources[0]}

        return {
            "source": {
                "$in": sources
            }
        }

    except Exception as e:
        logger.exception("Evaluation source filter creation failed")
        raise RAGPipelineException(str(e)) from e


def evaluate_single_question(
    qa_item,
    top_k=5,
    retrieval_top_k=10,
    user_id=None,
    where=None
):
    """Run retrieval, generation and evaluation for one question."""

    try:
        from src.reranking import retrieve_and_rerank
        from src.generation import generate_response
        from src.citations import process_citations

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be positive")

        if (
            not isinstance(retrieval_top_k, int)
            or isinstance(retrieval_top_k, bool)
            or retrieval_top_k <= 0
        ):
            raise ValueError("retrieval_top_k must be positive")

        question = qa_item["question"]
        question_type = qa_item["question_type"]
        question_id = qa_item["question_id"]

        logger.info(f"Evaluating question: {question_id}")

        retrieved_chunks = retrieve_and_rerank(
            query=question,
            top_k=top_k,
            retrieval_top_k=retrieval_top_k,
            user_id=user_id,
            where=where
        )

        if not isinstance(retrieved_chunks, list):
            raise ValueError("Retrieval must return a list")

        answer = generate_response(
            query=question,
            retrieved_chunks=retrieved_chunks,
            question_type=question_type
        )

        if not isinstance(answer, str):
            raise ValueError("Generation must return a string")

        citation_output = process_citations(
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )

        if not isinstance(citation_output, dict):
            raise ValueError("Invalid citation output")

        citation_verification = citation_output.get(
            "citation_verification",
            []
        )

        citation_accuracy = evaluate_citation_accuracy(
            citation_verification
        )
        citation_count = len(citation_verification)
        verified_citation_count = sum(
            1 for item in citation_verification
            if isinstance(item, dict) and item.get("verified") is True
        )

        abstained = is_abstention(answer)

        answer_metrics = None
        abstention_correct = None
        clarification_correct = None

        if question_type == "unanswerable":
            abstention_correct = abstained

        elif question_type == "ambiguous":
            clarification_correct = is_clarification(answer)

        else:
            answer_metrics = evaluate_answer_correctness(
                predicted_answer=answer,
                expected_answer=qa_item["expected_answer"]
            )

        relevant_chunk_ids = qa_item.get(
            "relevant_chunk_ids",
            []
        )

        retrieval_recall = calculate_recall_at_k(
            retrieved_chunks,
            relevant_chunk_ids
        )

        mrr = calculate_mrr(
            retrieved_chunks,
            relevant_chunk_ids
        )

        faithfulness = evaluate_lexical_faithfulness(
            answer,
            retrieved_chunks
        )

        confidence = citation_output.get("confidence_score")

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
        ):
            confidence = None

        return {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "predicted_answer": answer,
            "expected_answer": qa_item.get("expected_answer"),
            "answer_correctness": answer_metrics,
            "abstention_correct": abstention_correct,
            "clarification_correct": clarification_correct,
            "is_abstention": abstained,
            "retrieval_recall_at_k": retrieval_recall,
            "mean_reciprocal_rank": mrr,
            "citation_accuracy": citation_accuracy,
            "citation_count": citation_count,
            "verified_citation_count": verified_citation_count,
            "has_citations": citation_count > 0,
            "faithfulness_lexical_proxy": faithfulness,
            "confidence_indicator": confidence,
            "is_supported": citation_output.get("is_supported"),
            "retrieved_chunk_ids": get_retrieved_chunk_ids(
                retrieved_chunks
            ),
            "citation_verification": citation_verification
        }

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Single question evaluation failed")
        raise RAGPipelineException(str(e)) from e


def calculate_average(values):
    """Calculate an average while ignoring invalid values."""

    valid_values = [
        value
        for value in values
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    ]

    if not valid_values:
        return None

    return round(sum(valid_values) / len(valid_values), 4)


def calculate_aggregate_metrics(results):
    """Calculate metrics for the complete evaluation."""

    try:
        if not isinstance(results, list) or not results:
            raise ValueError("Evaluation results cannot be empty")

        successful_results = [
            result
            for result in results
            if result.get("status") == "success"
        ]

        failed_results = [
            result
            for result in results
            if result.get("status") == "failed"
        ]

        if not successful_results:
            raise ValueError("No successful evaluation results")

        answerable_results = [
            result
            for result in successful_results
            if result["question_type"] in {"direct", "multi_hop"}
        ]

        ambiguous_results = [
            result
            for result in successful_results
            if result["question_type"] == "ambiguous"
        ]

        unanswerable_results = [
            result
            for result in successful_results
            if result["question_type"] == "unanswerable"
        ]

        answer_metrics = [
            result["answer_correctness"]
            for result in answerable_results
            if result.get("answer_correctness") is not None
        ]

        return {
            "total_questions": len(results),
            "successful_questions": len(successful_results),
            "failed_questions": len(failed_results),
            "answerable_questions": len(answerable_results),
            "ambiguous_questions": len(ambiguous_results),
            "unanswerable_questions": len(unanswerable_results),

            "exact_match": calculate_average([
                item["exact_match"]
                for item in answer_metrics
            ]),

            "answer_token_precision": calculate_average([
                item["token_precision"]
                for item in answer_metrics
            ]),

            "answer_token_recall": calculate_average([
                item["token_recall"]
                for item in answer_metrics
            ]),

            "answer_token_f1": calculate_average([
                item["token_f1"]
                for item in answer_metrics
            ]),

            # None values (e.g. unanswerable questions with no gold chunks)
            # are excluded by calculate_average rather than treated as zero.
            "retrieval_recall_at_k": calculate_average([
                result.get("retrieval_recall_at_k")
                for result in successful_results
            ]),
            "retrieval_metric_question_count": sum(
                result.get("retrieval_recall_at_k") is not None
                for result in successful_results
            ),

            "mean_reciprocal_rank": calculate_average([
                result.get("mean_reciprocal_rank")
                for result in successful_results
            ]),

            # Micro-average over citations actually emitted. Responses without
            # citations are measured separately as citation coverage.
            "citation_accuracy": (
                round(
                    sum(result.get("verified_citation_count", 0) for result in successful_results)
                    / sum(result.get("citation_count", 0) for result in successful_results),
                    4,
                )
                if sum(result.get("citation_count", 0) for result in successful_results) > 0
                else None
            ),
            "total_citations": sum(
                result.get("citation_count", 0) for result in successful_results
            ),
            "verified_citations": sum(
                result.get("verified_citation_count", 0) for result in successful_results
            ),
            "citation_coverage": calculate_average([
                int(result.get("has_citations", False))
                for result in successful_results
            ]),

            "faithfulness_lexical_proxy": calculate_average([
                result["faithfulness_lexical_proxy"]
                for result in successful_results
            ]),

            "average_confidence_indicator": calculate_average([
                result["confidence_indicator"]
                for result in successful_results
            ]),

            "unanswerable_abstention_accuracy": calculate_average([
                int(result["abstention_correct"])
                for result in unanswerable_results
                if result.get("abstention_correct") is not None
            ]),

            "ambiguous_clarification_accuracy": calculate_average([
                int(result["clarification_correct"])
                for result in ambiguous_results
                if result.get("clarification_correct") is not None
            ])
        }

    except Exception as e:
        logger.exception("Aggregate metric calculation failed")
        raise RAGPipelineException(str(e)) from e


def compare_chunking_strategies(strategy_results):
    """Compare evaluation metrics across chunking strategies."""

    try:
        if not isinstance(strategy_results, dict) or not strategy_results:
            raise ValueError("Invalid strategy results")

        required_metrics = [
            "answer_token_f1",
            "retrieval_recall_at_k",
            "mean_reciprocal_rank",
            "citation_accuracy",
            "faithfulness_lexical_proxy"
        ]

        comparison = {}

        for strategy, metrics in strategy_results.items():
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"Invalid metrics for strategy: {strategy}"
                )

            comparison[strategy] = {}

            for metric in required_metrics:
                value = metrics.get(metric)

                if value is None:
                    comparison[strategy][metric] = None
                    continue

                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError(
                        f"Invalid {metric} for {strategy}"
                    )

                comparison[strategy][metric] = round(
                    float(value),
                    4
                )

        return comparison

    except Exception as e:
        logger.exception("Chunking strategy comparison failed")
        raise RAGPipelineException(str(e)) from e


def save_evaluation_report(report):
    """Save the evaluation report as JSON."""

    try:
        EVALUATION_REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_path = EVALUATION_REPORT_PATH.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

            file.flush()

        temporary_path.replace(EVALUATION_REPORT_PATH)

        logger.info(
            f"Evaluation report saved: {EVALUATION_REPORT_PATH}"
        )

        return EVALUATION_REPORT_PATH

    except Exception as e:
        logger.exception("Failed to save evaluation report")
        raise RAGPipelineException(str(e)) from e


def evaluate_dataset(
    dataset,
    top_k=5,
    retrieval_top_k=10,
    user_id=None,
    source_filter=None
):
    """Evaluate all questions and save the results."""

    try:
        normalized_dataset = normalize_qa_dataset(dataset)

        validate_qa_dataset(normalized_dataset)

        where = build_evaluation_filter(source_filter)

        results = []

        logger.info(
            f"Starting evaluation for {len(normalized_dataset)} questions"
        )

        for qa_item in normalized_dataset:
            try:
                result = evaluate_single_question(
                    qa_item=qa_item,
                    top_k=top_k,
                    retrieval_top_k=retrieval_top_k,
                    user_id=user_id,
                    where=where
                )

                result["status"] = "success"
                results.append(result)

            except Exception as e:
                logger.exception(
                    f"Evaluation failed for {qa_item['question_id']}"
                )

                results.append({
                    "question_id": qa_item["question_id"],
                    "question": qa_item["question"],
                    "question_type": qa_item["question_type"],
                    "status": "failed",
                    "error": str(e)
                })

        aggregate_metrics = calculate_aggregate_metrics(results)

        report = {
            "evaluation_timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "evaluation_configuration": {
                "top_k": top_k,
                "retrieval_top_k": retrieval_top_k,
                "user_id": user_id,
                "source_filter": (
                    source_filter
                    if source_filter is not None
                    else EVALUATION_SOURCES
                )
            },

            "aggregate_metrics": aggregate_metrics,
            "individual_results": results
        }

        save_evaluation_report(report)

        logger.info("Complete dataset evaluation finished")

        return report

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Dataset evaluation failed")
        raise RAGPipelineException(str(e)) from e


def main():
    """Run evaluation from the command line."""

    try:
        dataset = load_qa_dataset()

        report = evaluate_dataset(
            dataset=dataset,
            top_k=5,
            retrieval_top_k=10,
            source_filter=EVALUATION_SOURCES
        )

        print("\nEvaluation Summary")
        print("------------------")

        for metric, value in report["aggregate_metrics"].items():
            print(f"{metric}: {value}")

        print(
            f"\nReport saved to: {EVALUATION_REPORT_PATH}"
        )

    except RAGPipelineException as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in evaluation module")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

