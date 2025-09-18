import json
from pathlib import Path
import pytest
import sqlite3

from langchain.schema.document import Document

from ai_math_tutor.config import MISSING_PAGE_PLACEHOLDER
from ai_math_tutor.chunk_and_embed import create_or_update_content_database,  load_documents_from_json

def test_create_content_database(tmp_path):
    """Tests that the SQLite DB is created and populated correctly."""

    test_db_path = tmp_path / "test_book_content.db"
    
    # Create  LangChain Document objects
    docs = [Document(page_content=f"Page {num} content", metadata={"page_num": num}) for num in range(1, 4)]
    book_id = "test_book"

    create_or_update_content_database(docs, book_id, db_path=str(test_db_path))

    # Check that the database file was actually created
    assert test_db_path.exists()
    
    # Connect to the temporary database and verify its contents
    conn = sqlite3.connect(str(test_db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM pages WHERE book_id = ? AND page_number = ?", (book_id, 1))
    result = cursor.fetchone()
    conn.close()
    
    assert result is not None
    assert result[0] == "Page 1 content"


def test_load_documents_from_json_happy_path(tmp_path: Path):
    """
    Tests that the function correctly loads and parses a valid JSON file
    and converts it into a list of Document objects.
    """

    json_content = [
        {"page_num": 1, "page_content": "This is the content for page one."},
        {"page_num": 5, "page_content": "This is the content for page five."},
    ]
    
    test_file_path = tmp_path / "test_data.json"
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f)
        
    expected_documents = [
        Document(page_content="This is the content for page one.", metadata={"page_num": 1}),
        Document(page_content="This is the content for page five.", metadata={"page_num": 5}),
    ]

    result_documents = load_documents_from_json(str(test_file_path))

    assert result_documents == expected_documents



def test_load_documents_from_json_file_not_found(tmp_path: Path):
    """
    Tests that the function correctly raises a FileNotFoundError when the
    JSON file does not exist.
    """
    non_existent_path = tmp_path / "i_do_not_exist.json"

    with pytest.raises(FileNotFoundError) as excinfo:
        load_documents_from_json(str(non_existent_path))
    
    assert "File not found" in str(excinfo.value)


def test_load_documents_from_json_empty_file(tmp_path: Path):
    """
    Tests that the function correctly returns an empty list when the
    JSON file contains an empty list.
    """

    test_file_path = tmp_path / "empty_data.json"
    test_file_path.write_text("[]") 

    result_documents = load_documents_from_json(str(test_file_path))

    assert result_documents == []

def test_load_documents_from_json_missing_keys(tmp_path: Path):
    """
    Tests that the function raises a KeyError if the JSON objects
    are missing the required 'page_content' or 'page_num' keys.
    """

    json_content = [
        {"page_content": "This page is missing its number."}
    ]
    test_file_path = tmp_path / "missing_keys.json"
    with open(test_file_path, "w") as f:
        json.dump(json_content, f)

  
    with pytest.raises(KeyError):
        load_documents_from_json(str(test_file_path))


def test_load_documents_from_json_nonstring_value(tmp_path: Path):
    """
    Tests that the function correctly handles null 
    or other non-string page_content values.
    """
    json_content = [
        {"page_num": 1,
        "page_content": None,
        }
    ]
    test_file_path = tmp_path / "null_value.json"
    with open(test_file_path, "w") as f:
        json.dump(json_content, f)

    print
    result_documents = load_documents_from_json(str(test_file_path))

    assert result_documents[0].page_content == MISSING_PAGE_PLACEHOLDER
