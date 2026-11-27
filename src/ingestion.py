from pathlib import Path
import sys
import shutil
import re
from io import BytesIO

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from bs4 import BeautifulSoup
from pptx import Presentation
import pytesseract
import fitz
from PIL import Image

from src.config import (
    SUPPORTED_DOCUMENT_TYPES,
    MAX_FILE_SIZE_MB,
    TESSERACT_CMD
)
from src.logger import logger
from src.exception import RAGPipelineException


if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")


def extract_ocr_text(image):
    candidates = []

    for page_segmentation_mode in (3, 6, 11):
        try:
            data = pytesseract.image_to_data(
                image,
                config=f"--psm {page_segmentation_mode}",
                output_type=pytesseract.Output.DICT
            )

            lines = {}
            words = []

            for index, raw_word in enumerate(data.get("text", [])):
                word = raw_word.strip()
                try:
                    confidence = float(data["conf"][index])
                except (KeyError, IndexError, TypeError, ValueError):
                    confidence = -1

                if not word or confidence < 15:
                    continue

                key = (
                    data["block_num"][index],
                    data["par_num"][index],
                    data["line_num"][index]
                )
                position = (
                    data["top"][index],
                    data["left"][index],
                    word,
                    confidence
                )
                lines.setdefault(key, []).append(position)
                words.append((word, confidence))

            ordered_lines = sorted(
                lines.values(),
                key=lambda line: (
                    min(item[0] for item in line),
                    min(item[1] for item in line)
                )
            )
            text = "\n".join(
                " ".join(item[2] for item in sorted(line, key=lambda item: item[1]))
                for line in ordered_lines
            ).strip()

            if text:
                average_confidence = sum(
                    confidence for _, confidence in words
                ) / len(words)
                score = average_confidence + min(len(words), 100) * 0.05
                candidates.append((score, text, page_segmentation_mode))

        except Exception as ocr_error:
            logger.warning(
                "OCR layout mode {} failed: {}",
                page_segmentation_mode,
                ocr_error
            )

    if candidates:
        score, text, selected_mode = max(candidates, key=lambda item: item[0])
        logger.info(
            "Selected OCR layout mode {} with score {:.1f}",
            selected_mode,
            score
        )
        return text

    for page_segmentation_mode in (3, 11):
        text = pytesseract.image_to_string(
            image,
            config=f"--psm {page_segmentation_mode}"
        ).strip()
        if text:
            return text

    return ""


# Validate document existence, format and file size
def validate_document(file_path: str) -> Path:
    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a valid file: {path}")

        if path.suffix.lower() not in SUPPORTED_DOCUMENT_TYPES:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # Convert bytes to MB
        file_size_mb = path.stat().st_size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File size exceeds the limit of {MAX_FILE_SIZE_MB} MB"
            )

        logger.info(f"Document validated successfully: {path.name}")

        return path

    except Exception as e:
        logger.exception("Document validation failed!")
        raise RAGPipelineException(str(e)) from e


# Extract text from various document types
def extract_text(file_path: str) -> list[dict]:

    try:
        path = validate_document(file_path)
        extension = path.suffix.lower()

        extracted_documents = []

        # Extract text page by page from PDF, using OCR when needed
        if extension == ".pdf":

            pdf_document = fitz.open(str(path))

            try:
                for page_number, page in enumerate(
                    pdf_document, start=1
                ):

                    text = page.get_text().strip()
                    extraction_method = "native"

                    # Use OCR when native extraction finds little text
                    if len(text) < 40:

                        logger.info(
                            f"No native text found on page {page_number}. "
                            "Using OCR."
                        )

                        try:
                            pixmap = page.get_pixmap(
                                matrix=fitz.Matrix(4, 4),
                                alpha=False
                            )

                            image = Image.open(
                                BytesIO(pixmap.tobytes("png"))
                            )

                            ocr_text = extract_ocr_text(image)

                            if len(ocr_text) > len(text):
                                text = ocr_text
                                extraction_method = "ocr"

                        except Exception as ocr_error:
                            error_message = (
                                f"OCR failed for {path.name} page "
                                f"{page_number}. Install Tesseract OCR and "
                                "set TESSERACT_CMD in the project .env file. "
                                f"Details: {ocr_error}"
                            )
                            logger.error(error_message)
                            raise RAGPipelineException(error_message) from ocr_error

                    if text:

                        extracted_documents.append({
                            "text": text,
                            "metadata": {
                                "source": path.name,
                                "file_type": extension,
                                "page_number": page_number,
                                "extraction_method": extraction_method
                            }
                        })

                    else:
                        logger.warning(
                            f"No text extracted from page {page_number}"
                        )

            finally:
                pdf_document.close()

        # Extract paragraphs and tables from DOCX
        elif extension == ".docx":

            doc = Document(str(path))
            content_blocks = []

            for block in doc.element.body.iterchildren():
                if block.tag.endswith("}p"):
                    paragraph_text = Paragraph(block, doc).text.strip()
                    if paragraph_text:
                        content_blocks.append(paragraph_text)

                elif block.tag.endswith("}tbl"):
                    table = Table(block, doc)
                    for row in table.rows:
                        cells = [
                            " ".join(
                                paragraph.text.strip()
                                for paragraph in cell.paragraphs
                                if paragraph.text.strip()
                            )
                            for cell in row.cells
                        ]
                        row_text = " | ".join(
                            cell for cell in cells if cell
                        )
                        if row_text:
                            content_blocks.append(row_text)

            text = "\n".join(content_blocks)

            if text.strip():

                extracted_documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_type": extension
                    }
                })

        # Extract text from PPTX slides
        elif extension == ".pptx":

            presentation = Presentation(str(path))

            for slide_number, slide in enumerate(
                presentation.slides, start=1
            ):

                slide_parts = []
                for shape in slide.shapes:
                    if shape.has_text_frame and shape.text.strip():
                        slide_parts.append(shape.text.strip())
                    elif shape.has_table:
                        for row in shape.table.rows:
                            row_text = " | ".join(
                                cell.text.strip()
                                for cell in row.cells
                                if cell.text.strip()
                            )
                            if row_text:
                                slide_parts.append(row_text)

                slide_text = "\n".join(slide_parts)

                if slide_text.strip():

                    extracted_documents.append({
                        "text": slide_text,
                        "metadata": {
                            "source": path.name,
                            "file_type": extension,
                            "slide_number": slide_number
                        }
                    })

        # Extract readable text from HTML
        elif extension == ".html":

            html_content = path.read_text(encoding="utf-8")

            soup = BeautifulSoup(
                html_content,
                "html.parser"
            )

            # Remove unnecessary HTML elements
            for element in soup(["script", "style"]):
                element.decompose()

            for heading in soup.find_all(re.compile(r"^h[1-6]$")):
                level = int(heading.name[1])
                heading.insert(0, f"{'#' * level} ")

            # Extract text after removing unwanted elements
            text = soup.get_text(
                separator="\n",
                strip=True
            )

            if text.strip():

                extracted_documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_type": extension,
                        "extraction_method": "beautifulsoup"
                    }
                })

        # Extract text from Markdown and Plain Text
        elif extension in {".md", ".txt"}:

            text = path.read_text(encoding="utf-8")

            if text.strip():

                extracted_documents.append({
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "file_type": extension
                    }
                })

        logger.info(
            f"Text extraction completed for {path.name}. "
            f"({len(extracted_documents)} document units)"
        )

        return extracted_documents

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Text extraction failed!")
        raise RAGPipelineException(str(e)) from e


# Run document extraction from the command line
def main():

    try:
        if len(sys.argv) < 2:
            print("Usage: python -m src.ingestion <file_path>")
            return

        file_path = sys.argv[1]

        documents = extract_text(file_path)

        print(
            f"\nExtracted {len(documents)} document units."
        )

        for index, document in enumerate(
            documents, start=1
        ):

            print(f"\nDocument unit {index}")
            print(f"Metadata: {document['metadata']}")
            print(f"Text preview: {document['text'][:200]}")

    except RAGPipelineException as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"Ingestion failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in ingestion module")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()




