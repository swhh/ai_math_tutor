import json
import pathlib
import sqlite3
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch


from ai_math_tutor.config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    SAMPLE_BOOK,
    SAMPLE_OUTPUT_FILE_PATH,
    SQLITE_DB_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_or_update_content_database(documents: List[Document], book_id: str):
    """
    Creates and populates an SQLite database with the full page content,
    tagging each entry with a book_id.
    """
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS pages (
                book_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                PRIMARY KEY (book_id, page_number)
            )
            """
            )

            page_data = [
                (book_id, doc.metadata["page_num"], doc.page_content)
                for doc in documents
            ]

            cursor.executemany(
                "INSERT OR REPLACE INTO pages (book_id, page_number, content) VALUES (?, ?, ?)",
                page_data,
            )

    except sqlite3.Error as e:
        print(e)
        raise


def load_documents_from_json(json_path: str) -> List[Document]:

    filepath = pathlib.Path(json_path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found at '{json_path}'")
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    documents = [
        Document(
            page_content=page["page_content"], metadata={"page_num": page["page_num"]}
        )
        for page in json_data
    ]
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    markdown_splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = markdown_splitter.split_documents(documents)
    return chunks


def chunk_and_embed_pipeline(documents, collection_name):

    chunks = chunk_documents(documents)

    model_kwargs = {"device": DEVICE}
    encode_kwargs = {"normalize_embeddings": False}
    hf_embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=hf_embedding,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name,
    )
    return vector_store


if __name__ == "__main__":
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE}
    )
    vs = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=emb,
        collection_name=SAMPLE_BOOK,
    )

    documents = load_documents_from_json(SAMPLE_OUTPUT_FILE_PATH)
    print("collections:", [c.name for c in vs._client.list_collections()])
    print("count:", vs._collection.count())
