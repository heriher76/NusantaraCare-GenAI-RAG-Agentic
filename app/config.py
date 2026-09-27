import os
from dotenv import load_dotenv

load_dotenv()

NOTISPACE_API_KEY = os.getenv("NOTISPACE_API_KEY", "")
NOTISPACE_BASE_URL = os.getenv("NOTISPACE_BASE_URL", "https://api.notispaces.cloud/v1")
NOTISPACE_EMBED_MODEL = os.getenv("NOTISPACE_EMBED_MODEL", "notispace/ns-embed")
NOTISPACE_CHAT_MODEL = os.getenv("NOTISPACE_CHAT_MODEL", "notispace/ns-chat")

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
EMBED_BATCH_DELAY_SECONDS = float(os.getenv("EMBED_BATCH_DELAY_SECONDS", "0.5"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))
EMBED_RETRY_BASE_DELAY_SECONDS = float(os.getenv("EMBED_RETRY_BASE_DELAY_SECONDS", "1.0"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

RAW_DOC_PATH = os.getenv("RAW_DOC_PATH", "data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md")
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")

APP_NAME = "NusantaraCare RAG Assistant"
