import pytest
from langchain.schema.document import Document

from ai_math_tutor.utils import (
    format_docs,
    get_filename_without_extension,
    parse_page_numbers
)

# --- Tests for format_docs ---

def test_format_docs_with_multiple_documents():
    """Tests the happy path with a list of several Document objects."""
    # Arrange: Create some sample documents
    docs = [
        Document(page_content="This is page one.", metadata={"page_num": 1}),
        Document(page_content="This is page two.", metadata={"page_num": 2}),
    ]
    expected_output = "Page 1:\nThis is page one.\n\nPage 2:\nThis is page two."

    result = format_docs(docs)
    assert result == expected_output

def test_format_docs_with_empty_list():
    """Tests the edge case of an empty list of documents."""

    docs = []
    expected_output = ""

    result = format_docs(docs)
    assert result == expected_output

def test_format_docs_with_single_document():
    """Tests the simple case of a single document in the list."""
    docs = [Document(page_content="Only one page.", metadata={"page_num": 10})]
    expected_output = "Page 10:\nOnly one page."

    result = format_docs(docs)
    assert result == expected_output


# --- Tests for get_filename_without_extension ---

@pytest.mark.parametrize("input_path, expected_output", [
    ("data/my_book.pdf", "my_book"),
    ("data/archive.tar.gz", "archive.tar"),
    ("README", "README"),
    ("/home/user/documents/report.docx", "report"),
    ("C:\\Users\\user\\Desktop\\notes.txt", "notes"),
    ("", ""),
])
def test_get_filename_without_extension(input_path, expected_output):
    """Tests various file path formats."""

    result = get_filename_without_extension(input_path)
    assert result == expected_output


# --- Tests for parse_page_numbers ---

@pytest.mark.parametrize("input_list, expected_list", [
    (["45", "112-115", "10"], [10, 45, 112, 113, 114, 115]),
    ([], []),
    # Edge case: malformed ranges and non-numeric values should be gracefully ignored
    (["10-abc", "5", "xyz", "1-3"], [1, 2, 3, 5]),
    # Edge case: overlapping and duplicate numbers
    (["10-12", "11", "15-15", "10"], [10, 11, 12, 15]),
    (["5", "2", "100"], [2, 5, 100]),
    (["20-23"], [20, 21, 22, 23]),
])
def test_parse_page_numbers(input_list, expected_list):
    """Tests various page number string formats, including edge cases."""
    # Act
    result = parse_page_numbers(input_list)
    # Assert
    assert result == expected_list
