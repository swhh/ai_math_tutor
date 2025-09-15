import os
from typing import List

from langchain.schema.document import Document


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Page {doc.metadata['page_num']}:\n{doc.page_content}" for doc in docs
    )


def get_filename_without_extension(file_path: str) -> str:
    base_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension


if __name__ == "__main__":
    print(
        get_filename_without_extension(
            "/Users/seamusholland/ai_math_tutor/data/calculus_textbook.pdf"
        )
    )
