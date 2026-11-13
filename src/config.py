import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


#project directories
DATA_DIR = BASE_DIR / "data"
EVALUATION_DIR = BASE_DIR / "evaluation"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
BM25_DIR = STORAGE_DIR / "bm25"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

#AWS configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "raw/")
S3_PROCESSED_PREFIX = os.getenv("S3_PROCESSED_PREFIX", "processed/")

#Documentation configuration
SUPPORTED_DOCUMENT_TYPES = {
    ".pdf",".txt",".docx",".html",".md",".pptx"
}
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB


#embedding configuration
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"


#chunking configuration
CHUNK_SIZE = 1000  # Number of words per chunk
CHUNKING_STRATEGY = "recursive"  # Options: "recursive", "sliding_window"
CHUNK_OVERLAP = 200  # Number of overlapping words between chunks
SEMANTIC_SIMILARITY_THRESHOLD=0.75 #


#retrieval configuration
TOP_K=10  # Number of top documents to retrieve
DENSE_WEIGHT=0.7  # Weight for dense retrieval (Chroma)
SPARSE_WEIGHT=0.3  # Weight for sparse retrieval (BM25)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" #reranking model
RERANK_TOP_K = 5 #rerank top k documents


#generation configuration
LLM_MODEL=os.getenv("LLM_MODEL","qwen3:4b")
