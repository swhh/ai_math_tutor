from pathlib import Path
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory structure
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Database paths
SQLITE_DB_PATH = DATA_DIR / "book_content.db"

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

MISSING_PAGE_PLACEHOLDER = "--- CONTENT MISSING: This page could not be processed owing to a persistent error. ---"

# whether the AI assistant should return structured responses with AIAnswer or not
STRUCTURED_AI_ANSWERS = False


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DB_DIR.mkdir(exist_ok=True)


# Initialise directories on import
ensure_directories()
