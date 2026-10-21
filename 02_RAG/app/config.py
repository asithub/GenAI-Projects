import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHROMA_DIR = (PROJECT_ROOT / os.getenv("CHROMA_DIR", "data/vector_db_chroma")).resolve()
RAW_DOCS_DIR = (PROJECT_ROOT / os.getenv("RAW_DOCS_DIR", "data/raw")).resolve()

COLLECTION_NAME = "city_documents"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not configured")

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be positive")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE"
    )

if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1")

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)