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

# Sample data
SAMPLE_BOOK = "tao_complex_analysis_vol_two"

# File paths
SAMPLE_OUTPUT_FILE_PATH = DATA_DIR / f"{SAMPLE_BOOK}.json"

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DB_DIR.mkdir(exist_ok=True)


# Initialise directories on import
ensure_directories()
