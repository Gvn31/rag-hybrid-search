import json
import hashlib
from pathlib import Path
from collections import Counter

from src.vector_store import initialize_vector_store
from src.logger import logger
from src.exception import RAGPipelineException


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "exports"
)

OUTPUT_FILE = OUTPUT_DIR / "indexed_chunks.json"
INDEXING_MANIFEST_FILE = OUTPUT_DIR / "indexing_manifest.json"

# Evaluation corpus source documents
EVALUATION_SOURCES = [
    "api_documentation_sample.html",
    "knowledge_base_sample.md",
    "sample.pdf",
    "sample.pptx",
    "sample-document-medium.docx",
    "support_transcripts_sample.txt"
]


# Validate exported evaluation chunks
def validate_exported_chunks(chunks):

    try:
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("No evaluation chunks found")

        chunk_ids = set()

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Invalid chunk at index {index}"
                )

            if not chunk.get("chunk_id"):
                raise ValueError(
                    f"Missing chunk_id at index {index}"
                )

            if chunk["chunk_id"] in chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID: {chunk['chunk_id']}"
                )

            chunk_ids.add(chunk["chunk_id"])

            if chunk.get("source") not in EVALUATION_SOURCES:
                raise ValueError(
                    f"Unexpected source at index {index}"
                )

            if (
                not isinstance(chunk.get("text"), str)
                or not chunk["text"].strip()
            ):
                raise ValueError(
                    f"Invalid chunk text at index {index}"
                )

            if not isinstance(chunk.get("metadata"), dict):
                raise ValueError(
                    f"Invalid metadata at index {index}"
                )

        present_sources = {chunk.get("source") for chunk in chunks}
        missing_sources = set(EVALUATION_SOURCES) - present_sources
        if missing_sources:
            raise ValueError(
                "No indexed chunks found for source(s): "
                + ", ".join(sorted(missing_sources))
            )

        logger.info(
            f"Validated {len(chunks)} evaluation chunks"
        )

        return True

    except Exception as e:
        logger.exception("Evaluation chunk validation failed")
        raise RAGPipelineException(str(e)) from e


# Export only evaluation chunks from ChromaDB
def validate_indexing_manifest():
    """Reject exports when the latest indexing run is incomplete or stale."""

    try:
        if not INDEXING_MANIFEST_FILE.is_file():
            raise ValueError(
                "No indexing manifest found. Run the pipeline before exporting."
            )

        manifest = json.loads(
            INDEXING_MANIFEST_FILE.read_text(encoding="utf-8")
        )
        if manifest.get("evaluation_ready") is not True:
            incomplete = manifest.get("incomplete_sources", [])
            raise ValueError(
                "The latest pipeline run did not index every evaluation "
                "source successfully: " + ", ".join(incomplete)
            )

        raw_dir = PROJECT_ROOT / "data" / "raw"
        source_records = manifest.get("sources", {})
        for source in EVALUATION_SOURCES:
            source_path = raw_dir / source
            record = source_records.get(source, {})
            if record.get("status") != "success" or not source_path.is_file():
                raise ValueError(f"Source was not indexed successfully: {source}")
            current_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if current_hash != record.get("sha256"):
                raise ValueError(
                    f"Raw source changed after indexing; rerun the pipeline: {source}"
                )

        return True

    except Exception as e:
        logger.exception("Indexing manifest validation failed")
        raise RAGPipelineException(str(e)) from e


def export_indexed_chunks():

    try:
        logger.info("Starting evaluation chunk export")

        validate_indexing_manifest()

        # Initialize existing ChromaDB
        _, collection = initialize_vector_store()

        # Retrieve only evaluation documents
        records = collection.get(
            where={
                "source": {
                    "$in": EVALUATION_SOURCES
                }
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        ids = records.get("ids", [])
        documents = records.get("documents", [])
        metadatas = records.get("metadatas", [])

        if not (
            len(ids) == len(documents) == len(metadatas)
        ):
            raise ValueError(
                "ChromaDB returned inconsistent record counts"
            )

        exported_chunks = []

        for chroma_id, document, metadata in zip(
            ids,
            documents,
            metadatas
        ):

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Invalid metadata for record: {chroma_id}"
                )

            if not isinstance(document, str) or not document.strip():
                raise ValueError(
                    f"Empty document text: {chroma_id}"
                )

            chunk_id = metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    f"Missing chunk_id: {chroma_id}"
                )

            source = metadata.get("source")

            exported_chunks.append({
                "id": chroma_id,
                "chunk_id": chunk_id,
                "source": source,
                "text": document,
                "metadata": metadata
            })

        # Validate before writing
        validate_exported_chunks(exported_chunks)

        # Sort for reproducible exports
        exported_chunks.sort(
            key=lambda item: (
                item["source"],
                item["chunk_id"]
            )
        )

        # Generate source-wise counts
        source_counts = Counter(
            chunk["source"]
            for chunk in exported_chunks
        )

        export_data = {
            "total_chunks": len(exported_chunks),
            "source_counts": dict(source_counts),
            "chunks": exported_chunks
        }

        # Create output directory
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_file = OUTPUT_FILE.with_suffix(".tmp")

        # Save JSON safely
        with temporary_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                export_data,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

            file.flush()

        temporary_file.replace(OUTPUT_FILE)

        logger.info(
            f"Exported {len(exported_chunks)} evaluation chunks"
        )

        print("\nEvaluation Chunk Export Completed")
        print("----------------------------------")
        print("Exported chunks:", len(exported_chunks))

        print("\nSource-wise counts:")

        for source, count in source_counts.items():
            print(f"{source}: {count}")

        print("\nOutput file:", OUTPUT_FILE)

        return export_data

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Evaluation chunk export failed")
        raise RAGPipelineException(str(e)) from e


# Execute export independently
def main():

    try:
        export_indexed_chunks()

    except RAGPipelineException as e:
        logger.error(f"Export failed: {e}")
        print(f"Export failed: {e}")

    except Exception as e:
        logger.exception("Unexpected export error")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
