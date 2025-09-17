import os
from typing import List

from langchain.schema.document import Document


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(f"Page {doc.metadata['page_num']}:\n{doc.page_content}" for doc in docs)

def get_filename_without_extension(file_path: str) -> str:
    """Platform-agnostic function to get the filename without its extension"""
    if not file_path:
        return ""        
    # Normalise the path
    normalised_path = file_path.replace("\\", "/")

    base_name = os.path.basename(normalised_path)
    file_name_without_extension = os.path.splitext(base_name)[0]  
    return file_name_without_extension

def parse_page_numbers(pages: List[str]) -> List[int]:
    """Converts a list of page strings like ['45', '112-115'] to a list of ints [45, 112, 113, 114, 115]."""
    all_pages = set()
    for page_str in pages:
        if '-' in page_str:
            try:
                start, end = map(int, page_str.split('-'))
                all_pages.update(range(start, end + 1))
            except ValueError:
                continue # Skip malformed ranges
        else:
            try:
                all_pages.add(int(page_str))
            except ValueError:
                continue # Skip non-integer page numbers
    return sorted(list(all_pages))

