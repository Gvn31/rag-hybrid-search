import sys
import re
import unicodedata

from src.logger import logger
from src.exception import RAGPipelineException


#Normlize extracted text
def normalize_text(text:str)->str:
    try:
        if not isinstance(text,str):
            raise TypeError("Input text must be a string.")
        
        # Normalize Unicode characters
        text=unicodedata.normalize("NFKC",text)
        
        #Remove null characters and control characters
        text="".join(
            char for char in text
            if char=="\n" or char=="\t" or not 
            unicodedata.category(char).startswith("C")
        )
        
        #Normalize line endings
        text=text.replace("\r\n","\n").replace("\r","\n")
        
        #Remove excessive spaces and tabs
        text=re.sub(r"[\t]+","",text)
        
        #Remove leading and trailing whitespace
        text=text.strip()
        
        return text
    
    
    except Exception as e:
        logger.exception("Text Normalization Failed")
        raise RAGPipelineException(str(e))from e
    
    
    
#Preprocess extracted documents and preserve metadata
def preprocess_documents(documents:list[dict])->list[dict]:
    
    try:
        
        if not isinstance(documents,list):
            raise TypeError("Documents must be provided as a list")
        
        processed_documents=[]
        
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                raise TypeError(
                    f"Document at index {index} must be a dictionary"
                )

            if "text" not in document or "metadata" not in document:
                raise ValueError(
                    f"Document at index {index} must contain text and metadata"
                )

            # Normalize document text
            cleaned_text = normalize_text(document["text"])

            # Skip empty documents
            if not cleaned_text:
                logger.warning(
                    f"Skipping empty document at index {index}"
                )
                continue

            # Preserve original metadata
            processed_documents.append({
                "text": cleaned_text,
                "metadata": document["metadata"].copy()
            })

        logger.info(
            f"Preprocessing completed. "
            f"{len(processed_documents)} documents processed."
        )

        return processed_documents

    except RAGPipelineException:
        raise

    except Exception as e:
        logger.exception("Document preprocessing failed")
        raise RAGPipelineException(str(e)) from e


# Run preprocessing independently for testing
def main():
    try:
        sample_documents = [
            {
                "text": "  Employees    must follow\n\n\n security rules.  ",
                "metadata": {
                    "source": "sample.txt",
                    "file_type": ".txt"
                }
            }
        ]

        processed_documents = preprocess_documents(sample_documents)

        for index, document in enumerate(processed_documents, start=1):
            print(f"\nDocument {index}")
            print(f"Metadata: {document['metadata']}")
            print(f"Cleaned text: {document['text']}")

    except RAGPipelineException as e:
        logger.error(f"Preprocessing failed: {e}")
        print(f"Preprocessing failed: {e}")

    except Exception as e:
        logger.exception("Unexpected error in preprocessing module")
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

        