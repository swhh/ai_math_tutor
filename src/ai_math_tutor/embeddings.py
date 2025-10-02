import functools

from langchain_huggingface import HuggingFaceEmbeddings
import torch

from ai_math_tutor.config import EMBEDDING_MODEL_NAME

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# This is the single source of truth for the embedding model.

# NOTE: This @lru_cache creates a per-process singleton. Will need to replace this when using celery in production.
@functools.lru_cache(maxsize=1)
def get_embedding_model():
    """
    Creates and returns a single, shared instance of the embedding model.
    This is a per-process singleton.
    """
    model_kwargs = {"device": DEVICE}
    encode_kwargs = {"normalize_embeddings": False}

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs 
    )