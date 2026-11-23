import argparse
from pathlib import Path

from src.config import (
    SUPPORTED_DOCUMENT_TYPES,
    CHUNKING_STRATEGY
)

from src.logger import logger
from src.exception import RAGPipelineException

from src.ingestion import extract_text
from src.preprocessing import preprocess_documents
from src.chunking import chunk_documents
from src.embeddings import generate_embeddings
from src.indexing import index_embedded_chunks

from src.vector_store import initialize_vector_store
from src.bm25_store import get_index_count
from src.evaluation import (
    EVALUATION_REPORT_PATH,
    EVALUATION_SOURCES,
    evaluate_dataset,
    load_qa_dataset,
    compare_chunking_strategies,
    save_evaluation_report
)


# Set the project data directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Default corpus for general document uploads
DEFAULT_CORPUS_ID = "general"


# Validate corpus identifier
def validate_corpus_id(corpus_id):

    try:
        if (
            not isinstance(corpus_id, str)
            or not corpus_id.strip()
        ):
            raise ValueError(
                "corpus_id must be a non-empty string"
            )

        return corpus_id.strip()

    except Exception as e:
        logger.exception("Corpus ID validation failed")
        raise RAGPipelineException(str(e)) from e


# Process and index one document
def process_document(
    file_path,
    strategy=CHUNKING_STRATEGY,
    user_id=None,
    replace_existing=True,
    corpus_id=DEFAULT_CORPUS_ID,
    run_evaluation=True,
    evaluation_top_k=5,
    retrieval_top_k=50
):

    try:
        file_path = Path(file_path)

        corpus_id = validate_corpus_id(corpus_id)

        logger.info(
            f"Starting document pipeline: {file_path.name} | "
            f"Corpus: {corpus_id}"
        )

        # Step 1: Extract document text
        extracted_documents = extract_text(
            str(file_path)
        )

        if not extracted_documents:

            logger.warning(
                f"No text extracted from {file_path.name}"
            )

            return {
                "source": file_path.name,
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": "No text extracted"
            }

        # Step 2: Attach corpus metadata to extracted documents
        for document in extracted_documents:

            if not isinstance(document, dict):
                raise ValueError(
                    "Each extracted document must be a dictionary"
                )

            metadata = document.get("metadata")

            if not isinstance(metadata, dict):
                raise ValueError(
                    "Extracted document metadata must be a dictionary"
                )

            metadata["corpus_id"] = corpus_id

        # Step 3: Preprocess extracted documents
        processed_documents = preprocess_documents(
            extracted_documents
        )

        if not processed_documents:

            logger.warning(
                f"No usable text in {file_path.name}"
            )

            return {
                "source": file_path.name,
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": "No usable text after preprocessing"
            }

        # Step 4: Generate chunks
        chunks = chunk_documents(
            processed_documents,
            strategy=strategy
        )

        if not chunks:

            logger.warning(
                f"No chunks generated for {file_path.name}"
            )

            return {
                "source": file_path.name,
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": "No chunks generated"
            }

        # Step 5: Verify corpus metadata in generated chunks
        for chunk in chunks:

            metadata = chunk.get("metadata", {})

            if metadata.get("corpus_id") != corpus_id:
                raise ValueError(
                    "Corpus ID was not preserved during chunking"
                )

        # Step 6: Generate embeddings
        embedded_chunks = generate_embeddings(
            chunks
        )

        if not embedded_chunks:

            logger.warning(
                f"No embeddings generated for {file_path.name}"
            )

            return {
                "source": file_path.name,
                "corpus_id": corpus_id,
                "status": "skipped",
                "reason": "No embeddings generated"
            }

        # Step 7: Index the document in ChromaDB and BM25
        indexing_result = index_embedded_chunks(
            embedded_chunks=embedded_chunks,
            user_id=user_id,
            replace_document=replace_existing
        )

        # Add corpus information to the result
        indexing_result["corpus_id"] = corpus_id

        logger.info(
            f"Document pipeline completed: {file_path.name} | "
            f"Corpus: {corpus_id} | "
            f"Status: {indexing_result['status']}"
        )

        return indexing_result

    except RAGPipelineException:

        logger.exception(
            f"Document processing failed: {file_path.name}"
        )

        raise

    except Exception as e:

        logger.exception(
            f"Unexpected document processing error: {file_path.name}"
        )

        raise RAGPipelineException(str(e)) from e


# Process all supported documents in the raw data directory
def run_pipeline(
    raw_dir=RAW_DATA_DIR,
    strategy=CHUNKING_STRATEGY,
    user_id=None,
    replace_existing=True,
    corpus_id=DEFAULT_CORPUS_ID,
    run_evaluation=True,
    evaluation_top_k=5,
    retrieval_top_k=50,
    evaluation_report_path=None
):

    try:
        raw_dir = Path(raw_dir)

        corpus_id = validate_corpus_id(corpus_id)

        if not raw_dir.exists() or not raw_dir.is_dir():

            raise FileNotFoundError(
                f"Raw document directory not found: {raw_dir}"
            )

        if not isinstance(replace_existing, bool):

            raise ValueError(
                "replace_existing must be a boolean"
            )

        if user_id is not None:

            if (
                not isinstance(user_id, str)
                or not user_id.strip()
            ):
                raise ValueError(
                    "user_id must be a non-empty string"
                )

        # Find supported documents
        document_files = sorted(
            file_path
            for file_path in raw_dir.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_DOCUMENT_TYPES
            )
        )

        if not document_files:
            raise FileNotFoundError(
                f"No supported documents found in {raw_dir}"
            )

        logger.info(
            f"Found {len(document_files)} documents in {raw_dir} | "
            f"Corpus: {corpus_id}"
        )

        results = []

        # Process documents one by one
        for file_path in document_files:

            logger.info(
                f"Processing: {file_path.name}"
            )

            result = process_document(
                file_path=file_path,
                strategy=strategy,
                user_id=user_id,
                replace_existing=replace_existing,
                corpus_id=corpus_id
            )

            results.append(result)

        # Calculate processing statistics
        successful = sum(
            1
            for result in results
            if result.get("status") == "success"
        )

        skipped = sum(
            1
            for result in results
            if result.get("status") in {
                "skipped",
                "all_duplicates"
            }
        )

        # Read final index counts
        _, collection = initialize_vector_store()
        chroma_count = collection.count()
        bm25_count = get_index_count()

        if successful == 0:
            raise RAGPipelineException(
                "No documents were indexed successfully; evaluation was not started."
            )

        evaluation_report = None

        # Evaluate the indexed documents against the complete QA dataset.
        if run_evaluation:
            logger.info("Starting end-to-end QA evaluation")
            dataset = load_qa_dataset()
            evaluation_report = evaluate_dataset(
                dataset=dataset,
                top_k=evaluation_top_k,
                retrieval_top_k=retrieval_top_k,
                user_id=user_id,
                source_filter=EVALUATION_SOURCES,
                report_path=evaluation_report_path
            )

        # Prepare the complete pipeline summary.
        summary = {
            "total_files": len(document_files),
            "successful": successful,
            "skipped": skipped,
            "corpus_id": corpus_id,
            "chroma_count": chroma_count,
            "bm25_count": bm25_count,
            "evaluation_completed": evaluation_report is not None,
            "evaluation_report": (
                str(evaluation_report.get("report_path", EVALUATION_REPORT_PATH))
                if evaluation_report else None
            ),
            "evaluation_metrics": (
                evaluation_report.get("aggregate_metrics")
                if evaluation_report else None
            ),
            "results": results
        }

        logger.info(
            f"Pipeline completed | "
            f"Files: {len(document_files)} | "
            f"Successful: {successful} | "
            f"Skipped: {skipped} | "
            f"Corpus: {corpus_id} | "
            f"ChromaDB: {chroma_count} | "
            f"BM25: {bm25_count}"
        )

        return summary

    except RAGPipelineException:
        raise

    except Exception as e:

        logger.exception(
            "RAG pipeline execution failed"
        )

        raise RAGPipelineException(str(e)) from e




# Evaluate every chunking strategy using the same QA dataset

def run_chunking_strategy_comparison(
    raw_dir=RAW_DATA_DIR,
    strategies=("fixed_size", "recursive", "semantic"),
    user_id=None,
    corpus_id=DEFAULT_CORPUS_ID,
    evaluation_top_k=5,
    retrieval_top_k=50
):

    try:
        supported_strategies = {"fixed_size", "recursive", "semantic"}
        if (
            not isinstance(strategies, (list, tuple))
            or len(strategies) != len(supported_strategies)
            or set(strategies) != supported_strategies
        ):
            raise ValueError(
                "Strategies must include fixed_size, recursive, and semantic exactly once"
            )

        strategy_results = {}
        report_paths = {}

        for strategy in strategies:
            strategy_report_path = EVALUATION_REPORT_PATH.with_name(
                f"evaluation_report_{strategy}.json"
            )
            logger.info("Running chunking comparison for {}", strategy)

            run_summary = run_pipeline(
                raw_dir=raw_dir,
                strategy=strategy,
                user_id=user_id,
                replace_existing=True,
                corpus_id=corpus_id,
                run_evaluation=True,
                evaluation_top_k=evaluation_top_k,
                retrieval_top_k=retrieval_top_k,
                evaluation_report_path=strategy_report_path
            )

            metrics = run_summary.get("evaluation_metrics")
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"No evaluation metrics returned for {strategy}"
                )

            strategy_results[strategy] = metrics
            report_paths[strategy] = str(strategy_report_path)

        comparison = compare_chunking_strategies(strategy_results)
        comparison_report = {
            "strategies": list(strategies),
            "retrieval_comparison_method": (
                "Source-level relevance is used because chunk IDs change by strategy."
            ),
            "reports": report_paths,
            "metrics": comparison
        }
        comparison_path = EVALUATION_REPORT_PATH.with_name(
            "chunking_strategy_comparison.json"
        )
        save_evaluation_report(comparison_report, comparison_path)

        run_pipeline(
            raw_dir=raw_dir,
            strategy=CHUNKING_STRATEGY,
            user_id=user_id,
            replace_existing=True,
            corpus_id=corpus_id,
            run_evaluation=False
        )

        logger.info(
            "Chunking strategy comparison saved: {}",
            comparison_path
        )
        return {
            "comparison_report": str(comparison_path),
            "reports": report_paths,
            "metrics": comparison,
            "active_strategy_restored": CHUNKING_STRATEGY
        }

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Chunking strategy comparison failed")
        raise RAGPipelineException(str(e)) from e


# Execute the complete pipeline
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete RAG document indexing pipeline"
        )
    )

    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DATA_DIR),
        help="Directory containing raw documents"
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "fixed_size",
            "recursive",
            "semantic"
        ],
        default=CHUNKING_STRATEGY,
        help="Document chunking strategy"
    )

    parser.add_argument(
        "--compare-chunking-strategies",
        action="store_true",
        help="Run evaluation with fixed-size, recursive, and semantic chunking"
    )

    parser.add_argument(
        "--user-id",
        default=None,
        help="User ID for document isolation"
    )

    parser.add_argument(
        "--corpus-id",
        default=DEFAULT_CORPUS_ID,
        help=(
            "Corpus identifier for document separation "
            "(default: general)"
        )
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not replace existing document versions"
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Run ingestion and indexing without QA evaluation"
    )

    parser.add_argument(
        "--evaluation-top-k",
        type=int,
        default=5,
        help="Number of final retrieved chunks used during evaluation"
    )

    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=50,
        help="Candidate chunks retrieved before reranking during evaluation"
    )

    args = parser.parse_args()

    try:

        print(
            "\nStarting RAG document indexing pipeline...\n"
        )

        if args.compare_chunking_strategies:
            comparison = run_chunking_strategy_comparison(
                raw_dir=args.raw_dir,
                user_id=args.user_id,
                corpus_id=args.corpus_id,
                evaluation_top_k=args.evaluation_top_k,
                retrieval_top_k=args.retrieval_top_k
            )
            print("\nChunking Strategy Comparison")
            print("-" * 40)
            for strategy, metrics in comparison["metrics"].items():
                print(f"{strategy}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")
            print(f"\nComparison report: {comparison['comparison_report']}")
            return

        summary = run_pipeline(
            raw_dir=args.raw_dir,
            strategy=args.strategy,
            user_id=args.user_id,
            replace_existing=not args.keep_existing,
            corpus_id=args.corpus_id,
            run_evaluation=not args.skip_evaluation,
            evaluation_top_k=args.evaluation_top_k,
            retrieval_top_k=args.retrieval_top_k
        )

        print("\nPipeline Summary")
        print("-" * 40)

        for key, value in summary.items():

            if key != "results":
                print(f"{key}: {value}")

        print("\nIndividual Document Results")

        for result in summary["results"]:

            print(
                f"{result.get('source', 'Unknown')}: "
                f"{result.get('status', 'unknown')} | "
                f"Corpus: {result.get('corpus_id', 'unknown')}"
            )

        print(
            "\nPipeline execution completed."
        )

        if summary.get("evaluation_completed"):
            print("\nEvaluation Metrics")
            print("-" * 40)
            for metric, value in summary["evaluation_metrics"].items():
                print(f"{metric}: {value}")
            print(f"\nEvaluation report: {summary['evaluation_report']}")

    except RAGPipelineException as e:

        logger.error(
            f"Pipeline failed: {e}"
        )

        print(
            f"\nPipeline failed: {e}"
        )

    except Exception as e:

        logger.exception(
            "Unexpected pipeline execution error"
        )

        print(
            f"\nUnexpected error: {e}"
        )


if __name__ == "__main__":
    main()

