  
import sys
import re
import hashlib
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKING_STRATEGY,
    SEMANTIC_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL
)

from src.logger import logger
from src.exception import RAGPipelineException


# Load the embedding model only when semantic chunking is used
_embedding_model = None


# Load the sentence transformer model
def get_embedding_model():
    global _embedding_model

    try:
        if _embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        return _embedding_model

    except Exception as e:
        logger.exception("Failed to load embedding model")
        raise RAGPipelineException(str(e))


# Find the nearest section heading before a given text position
def get_section_heading(text, position):
    try:
        heading_pattern = re.compile(
            r"^(?:#{1,6}\s+.+|(?:\d+(?:\.\d+)*[.)]?\s+).+)$",
            re.MULTILINE
        )

        headings = list(heading_pattern.finditer(text[:position + 1]))

        if headings:
            return headings[-1].group().strip()

        return ""

    except Exception as e:
        logger.exception("Failed to extract section heading")
        raise RAGPipelineException(str(e))


# Split text into fixed-size chunks with overlap
def fixed_size_chunking(text):
    try:
        chunks = []

        if not text or not text.strip():
            return chunks

        text_length = len(text)
        start = 0

        while start < text_length:
            end = min(start + CHUNK_SIZE, text_length)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_index": start,
                    "end_index": end,
                    "section_heading": get_section_heading(text, start)
                })

            # Move forward while maintaining overlap
            if end >= text_length:
                break

            next_start = end - CHUNK_OVERLAP

            if next_start <= start:
                raise ValueError(
                    "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
                )

            start = next_start

        logger.info(f"Fixed-size chunking created {len(chunks)} chunks")

        return chunks

    except Exception as e:
        logger.exception("Fixed-size chunking failed")
        raise RAGPipelineException(str(e))


# Split text recursively using separators
def recursive_chunking(text):
    try:
        if not text or not text.strip():
            return []

        # Define the order of separators for recursive splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
            length_function=len,
            add_start_index=True
        )

        # Generate chunks with their original text positions
        split_documents = text_splitter.create_documents([text])

        chunks = []

        for document in split_documents:
            chunk_text = document.page_content.strip()
            start_index = document.metadata.get("start_index", 0)

            if not chunk_text:
                continue

            # Find the actual end position in the original text
            end_index = start_index + len(document.page_content)

            chunks.append({
                "text": chunk_text,
                "start_index": start_index,
                "end_index": end_index,
                "section_heading": get_section_heading(text, start_index)
            })

        logger.info(f"Recursive chunking created {len(chunks)} chunks")

        return chunks

    except Exception as e:
        logger.exception("Recursive chunking failed")
        raise RAGPipelineException(str(e))


# Split text into sentences while preserving original positions
def split_into_sentences(text):
    try:
        sentence_pattern = re.compile(
            r".+?(?:[.!?](?=\s|$)|\n+|$)",
            re.DOTALL
        )

        sentences = []

        for match in sentence_pattern.finditer(text):
            sentence_text = match.group().strip()

            if sentence_text:
                # Locate the actual sentence boundaries in the original text
                original_text = match.group()
                leading_spaces = len(original_text) - len(
                    original_text.lstrip()
                )

                trailing_spaces = len(original_text) - len(
                    original_text.rstrip()
                )

                start_index = match.start() + leading_spaces
                end_index = match.end() - trailing_spaces

                sentences.append({
                    "text": sentence_text,
                    "start_index": start_index,
                    "end_index": end_index
                })

        return sentences

    except Exception as e:
        logger.exception("Sentence splitting failed")
        raise RAGPipelineException(str(e))


# Group sentences based on semantic similarity
def semantic_chunking(text):
    try:
        if not text or not text.strip():
            return []

        # Split the document into individual sentences
        sentences = split_into_sentences(text)

        if not sentences:
            return []

        # Load the embedding model
        model = get_embedding_model()

        sentence_texts = [
            sentence["text"] for sentence in sentences
        ]

        # Convert sentences into normalized vector embeddings
        embeddings = model.encode(
            sentence_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        chunks = []
        current_group = [0]

        # Compare each sentence with the previous sentence
        for index in range(1, len(sentences)):
            previous_embedding = embeddings[index - 1]
            current_embedding = embeddings[index]

            similarity = float(
                np.dot(previous_embedding, current_embedding)
            )

            current_start = sentences[current_group[0]]["start_index"]
            current_end = sentences[index]["end_index"]

            current_length = current_end - current_start

            # Merge sentences when they are semantically related
            # and the maximum chunk size is not exceeded
            if (
                similarity >= SEMANTIC_SIMILARITY_THRESHOLD
                and current_length <= CHUNK_SIZE
            ):
                current_group.append(index)

            else:
                # Create a chunk from the current sentence group
                group_start = sentences[current_group[0]]["start_index"]
                group_end = sentences[current_group[-1]]["end_index"]

                chunks.append({
                    "text": text[group_start:group_end].strip(),
                    "start_index": group_start,
                    "end_index": group_end,
                    "section_heading": get_section_heading(
                        text, group_start
                    )
                })

                current_group = [index]

        # Add the final sentence group
        if current_group:
            group_start = sentences[current_group[0]]["start_index"]
            group_end = sentences[current_group[-1]]["end_index"]

            chunks.append({
                "text": text[group_start:group_end].strip(),
                "start_index": group_start,
                "end_index": group_end,
                "section_heading": get_section_heading(
                    text, group_start
                )
            })

        logger.info(f"Semantic chunking created {len(chunks)} chunks")

        return chunks

    except Exception as e:
        logger.exception("Semantic chunking failed")
        raise RAGPipelineException(str(e))


# Apply the selected chunking strategy to all documents
def chunk_documents(documents, strategy=CHUNKING_STRATEGY):
    try:
        if not isinstance(documents, list):
            raise ValueError("Documents must be provided as a list")

        # Map each strategy to its corresponding chunking function
        chunking_functions = {
            "fixed_size": fixed_size_chunking,
            "recursive": recursive_chunking,
            "semantic": semantic_chunking
        }

        if strategy not in chunking_functions:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        chunking_function = chunking_functions[strategy]

        all_chunks = []

        # Process each document individually
        for document in documents:

            if not isinstance(document, dict):
                raise ValueError("Each document must be a dictionary")

            if "text" not in document or "metadata" not in document:
                raise ValueError(
                    "Each document must contain text and metadata"
                )

            text = document["text"]
            metadata = document["metadata"]

            if not isinstance(text, str):
                raise ValueError("Document text must be a string")

            if not isinstance(metadata, dict):
                raise ValueError("Document metadata must be a dictionary")

            if not text.strip():
                logger.warning("Skipping empty document")
                continue

            # Generate chunks using the selected strategy
            document_chunks = chunking_function(text)

            # Attach document metadata and unique chunk identifiers
            for index, chunk in enumerate(document_chunks):

                chunk_text = chunk["text"]

                chunk_id = hashlib.sha256(
                    (
                        str(metadata.get("source", "unknown"))
                        + str(chunk["start_index"])
                        + chunk_text
                    ).encode("utf-8")
                ).hexdigest()

                chunk_metadata = metadata.copy()

                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunking_strategy": strategy,
                    "section_heading": chunk["section_heading"],
                    "char_count": len(chunk_text),
                    "start_index": chunk["start_index"],
                    "end_index": chunk["end_index"]
                })

                all_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

        logger.info(
            f"Total chunks created: {len(all_chunks)} "
            f"using {strategy} strategy"
        )

        return all_chunks

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Document chunking failed")
        raise RAGPipelineException(str(e))


# Test all three chunking strategies independently
def main():
    try:
        sample_documents = [
            {
                "text": (
                    "# Introduction\n"
                    "Artificial intelligence helps computers perform "
                    "tasks that normally require human intelligence.\n\n"

                    "## Machine Learning\n"
                    "Machine learning enables systems to learn patterns "
                    "from data and make predictions.\n\n"

                    "## Deep Learning\n"
                    "Deep learning uses neural networks with multiple "
                    "layers to learn complex data representations.\n\n"

                    "## Natural Language Processing\n"
                    "Natural language processing enables computers to "
                    "understand and process human language."
                ),
                "metadata": {
                    "source": "sample_document.txt",
                    "file_type": ".txt"
                }
            }
        ]

        # Test each available chunking strategy
        for strategy in ["fixed_size", "recursive", "semantic"]:

            print(f"\nTesting strategy: {strategy}")

            chunks = chunk_documents(
                sample_documents,
                strategy=strategy
            )

            # Display chunk content and metadata
            for chunk in chunks:
                print(
                    f"\nChunk ID: "
                    f"{chunk['metadata']['chunk_id'][:12]}"
                )

                print(f"Metadata: {chunk['metadata']}")
                print(f"Text: {chunk['text']}")

    except RAGPipelineException as e:
        logger.error(f"Chunking failed: {e}")
        print(f"Chunking failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in chunking module")
        print(f"Unexpected error: {e}")


# Run the test when this file is executed directly
if __name__ == "__main__":
    main()
