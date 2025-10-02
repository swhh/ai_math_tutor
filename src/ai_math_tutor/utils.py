import aiofiles
import os
from pathlib import Path
from typing import List
import uuid

from fastapi import UploadFile
from langchain.schema.document import Document



def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Page {doc.metadata['page_num']}:\n{doc.page_content}" for doc in docs
    )


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
        if "-" in page_str:
            try:
                start, end = map(int, page_str.split("-"))
                all_pages.update(range(start, end + 1))
            except ValueError:
                continue  # Skip malformed ranges
        else:
            try:
                all_pages.add(int(page_str))
            except ValueError:
                continue  # Skip non-integer page numbers
    return sorted(list(all_pages))


async def save_upload_file_async(upload_file: UploadFile, destination: Path) -> None:
    """
    Asynchronously saves an uploaded file to a destination path.
    """
    try:
        async with aiofiles.open(destination, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # Read in 1MB chunks
                await out_file.write(content)
    finally:
        await upload_file.close()


def generate_collection_name(filename):
    original_name = Path(filename).name
    title = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    collection_name = f"{title}__{uuid.uuid4().hex}{suffix}"
    return collection_name
