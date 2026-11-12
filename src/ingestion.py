from pathlib import Path
import sys
from io import BytesIO

from pypdf import PdfReader
from docx import Document 
from bs4 import BeautifulSoup
from pptx import Presentation
from markdown import markdown
import pytesseract
import fitz
from PIL import Image

from src.config import SUPPORTED_DOCUMENT_TYPES, MAX_FILE_SIZE_MB
from src.logger import logger
from src.exception import RAGPipelineException


pytesseract.pytesseract.tesseract_cmd = (
    r"D:\tesseract-OCR\tesseract.exe"
)


#Validate document existence, format and file size.
def validate_document(file_path: str) -> Path:
        """Validate document existence, format and file size."""
        try:
            path=Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}.")
            
            if not path.is_file():
                raise ValueError(f"Not a valid file: {path}.")
            
            if path.suffix.lower() not in SUPPORTED_DOCUMENT_TYPES:
                raise ValueError(f"Unsupported file type: {path.suffix}.")
            
            file_size_mb = path.stat().st_size / (1024 * 1024)  # Convert bytes to MB
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(f"File size exceeds size of {MAX_FILE_SIZE_MB} MB.")
            
            logger.info(f"Document validated successfully: {path.name}.")
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
        
        #extract text page by page from PDF ,also using OCR when needed
        if extension == ".pdf":
            pdf_document = fitz.open(str(path))
            try:
                for page_number, page in enumerate(pdf_document,start=1):
                    text = page.get_text().strip()
                    extraction_method = "native"
                    
                    #Use OCR if native text extraction returns empty
                    if not text:
                        logger.info(f"No native text found on page {page_number}. Using OCR.")
                        
                        pixmap=page.get_pixmap(matrix=fitz.Matrix(4,4),alpha=False)
                        image=Image.open(BytesIO(pixmap.tobytes("png")))
                        text = pytesseract.image_to_string(image, config="--psm 11").strip()
                        extraction_method = "ocr"
                        
                    if text:
                        extracted_documents.append({
                            "text": text,
                            "metadata":{
                                "source": path.name,
                                "file_type": extension,
                                "page_number": page_number,
                                "extraction_method": extraction_method   
                            }
                        })
                        
                    else:
                        logger.warning(f"No text extracted from page {page_number} .")
                        
            finally:
                pdf_document.close()   
                    
        #extract paragraphs and tables from DOCX
        elif extension == ".docx":
            doc = Document(str(path))
            
            paragraphs=[
                paragraph.text
                for paragraph in doc.paragraphs
                if paragraph.text.strip()
            ]
            
            table_text=[
                cell.text
                for table in doc.tables
                for row in table.rows
                for cell in row.cells
                if cell.text.strip()
            ]
            
            text="\n".join(paragraphs + table_text)
            
            if text.strip():
                extracted_documents.append({
                    "text": text,
                    "metadata":{
                        "source": path.name,
                        "file_type": extension
                    }
                })
                
        
        #extract text slide from PPTX
        elif extension == ".pptx":
            presentation=Presentation(str(path))
            
            for slide_number, slide in enumerate(
                presentation.slides,start=1
            ):
                slide_text="\n".join(
                    shape.text
                    for shape in slide.shapes
                    if shape.has_text_frame and shape.text.strip()
                )
                
                if slide_text.strip():
                    extracted_documents.append({
                        "text": slide_text,
                        "metadata":{
                            "source": path.name,
                            "file_type": extension,
                            "slide_number": slide_number
                        }
                    })
                    
        #extract readable text from HTML
        elif extension == ".html":
           html_content=path.read_text(encoding="utf-8")
           soup=BeautifulSoup(html_content,"html.parser")
           
           for element in soup(["script","style"]):
               element.decompose()
               
               text=soup.get_text(separator="\n",strip=True)
               
               if text.strip():
                     extracted_documents.append({
                          "text": text,
                          "metadata":{
                            "source": path.name,
                            "file_type": extension
                          }
                     })
                     
        #extract text from Markdown and Plain Text
        elif extension  in {".md",".txt"}:
            text=path.read_text(encoding="utf-8")
            
            if text.strip():
                extracted_documents.append({
                    "text": text,
                    "metadata":{
                        "source": path.name,
                        "file_type": extension
                    }
                })
                
        logger.info(f"Text extraction completed for {path.name}."
                     f"({len(extracted_documents)} document units)")
        
        return extracted_documents
    
    except RAGPipelineException:
        raise
    except Exception as e:
        logger.exception("Text extraction failed!")
        raise RAGPipelineException(str(e)) from e
            
          
# Run documennt extraction from the command line
def main():
    try:
        if len(sys.argv) < 2:
            print("Usage: python ingestion.py <file_path>")
            return
        
        file_path=sys.argv[1]
        documents=extract_text(file_path)
        
        
        print(f"\nExtracted {len(documents)} document units.")

        for index, document in enumerate(documents, start=1):
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

           