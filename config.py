"""PDF tabanli retrieval hatti icin merkezi ayarlar."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma"

EMBEDDING_MODEL_NAME = "nomic-embed-text"
OLLAMA_MODEL_NAME = "gpt-oss:120b-cloud"

RETRIEVAL_TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
