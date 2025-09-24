import json
import pathlib
import sqlite3
from typing import List, Optional

from google.genai.errors import ServerError
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import torch

from ai_math_tutor.config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    MISSING_PAGE_PLACEHOLDER,
    SQLITE_DB_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_LLM = "gemini-1.5-flash"
MAX_LOOKBACK = 50


class IndexEntry(BaseModel):
    """Represents a single term and its associated pages from the book's index."""

    term: str = Field(
        ..., description="The specific keyword or concept from the index."
    )
    pages: List[str] = Field(
        ...,
        description="A list of page numbers or page ranges (e.g., ['45', '112-115']).",
    )


class BookIndex(BaseModel):
    """
    The full, parsed index of the book. Includes a flag to indicate
    if an index was successfully found and parsed.
    """

    index_found: bool = Field(
        ...,
        description="Set to true if the provided text is a book index and was parsed successfully, otherwise set to false.",
    )
    entries: Optional[List[IndexEntry]] = Field(
        default=None,
        description="The list of parsed index entries. This should be null if index_found is false.",
    )


def create_or_update_content_database(
    documents: List[Document], book_id: str, db_path=SQLITE_DB_PATH
):
    """
    Creates and populates an SQLite database with the full page content,
    tagging each entry with a book_id.
    """
    try:
        with sqlite3.connect(db_path) as conn:
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
        raise e


def load_documents_from_json(json_path: str) -> List[Document]:

    filepath = pathlib.Path(json_path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found at '{json_path}'")
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    documents = [
        Document(
            page_content=(
                page.get("page_content") or MISSING_PAGE_PLACEHOLDER
            ),  # in case llm returns null
            metadata={"page_num": page["page_num"]},
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


def extract_index(documents: List[Document]) -> BookIndex:
    total_page_num = len(documents)
    start_page = max(
        total_page_num - MAX_LOOKBACK, int(total_page_num * 0.9)
    )  # only look at last MAX_LOOKBACK pages

    index_text = "\n\n".join(
        f"--- Page {doc.metadata['page_num']} ---\n{doc.page_content}"
        for doc in documents
        if doc.metadata["page_num"] >= start_page
    )
    index_llm = init_chat_model(INDEX_LLM, model_provider="google_genai", temperature=0)
    structured_llm = index_llm.with_structured_output(BookIndex)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert document parser. Your task is to extract the keyword index from the provided text.
            - If the text clearly appears to be a book index, parse it into a list of terms and their associated page numbers. Set `index_found` to true.
            - If the text does NOT appear to be a book index, or if you are unable to parse it, you MUST set `index_found` to false and leave the `entries` field null.""",
            ),
            (
                "human",
                "Please parse the book index from the following text:\n\n{index_text}",
            ),
        ]
    )

    chain = prompt | structured_llm.with_retry(
        retry_if_exception_type=(ServerError,),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )
    try:
        book_index = chain.invoke({"index_text": index_text})
        return book_index
    except Exception as e:
        print(f"Book indexing failed: {e}")
    return BookIndex(index_found=False, entries=None)


def store_index(
    book_index: BookIndex, book_id: str, content_db_path: str = SQLITE_DB_PATH
):
    try:
        with sqlite3.connect(content_db_path) as conn:
            cursor = conn.cursor()

            # --- Create an FTS5 Virtual Table ---
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS book_index_fts
                USING fts5(term, pages_json, book_id UNINDEXED, tokenize = 'porter');
                """
            )

            # --- Create book index table ---
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS book_index (
                    book_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    pages_json TEXT NOT NULL,
                    PRIMARY KEY (book_id, term)
                )
                """
            )

            index_data = [
                (book_id, entry.term.lower(), json.dumps(entry.pages))
                for entry in book_index.entries
            ]
            cursor.executemany(
                "INSERT OR REPLACE INTO book_index VALUES (?, ?, ?)", index_data
            )

            # Populate the FTS table from the main table deleting existing entries for the book to prevent duplicates.
            cursor.execute("DELETE FROM book_index_fts WHERE book_id = ?", (book_id,))
            cursor.execute(
                """
                INSERT INTO book_index_fts (term, pages_json, book_id)
                SELECT term, pages_json, book_id FROM book_index WHERE book_id = ?
                """,
                (book_id,),
            )
    except Exception as e:
        raise e


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
