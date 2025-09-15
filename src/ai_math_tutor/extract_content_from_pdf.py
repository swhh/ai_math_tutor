import asyncio
import json
import os
import pathlib
from typing import Any, Dict, List

from google import genai
from google.genai import types
from google.genai.errors import ServerError
import pymupdf
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ai_math_tutor.config import PROJECT_ROOT

MODEL_ID = "gemini-2.5-flash"
GEMINI_PROMPT = """
You are an expert document analyst specializing in academic textbooks.
Your task is to convert the content of the provided page image into a clean, well-structured Markdown format.

Instructions:
1.  Preserve the semantic structure of the page. Use Markdown headers (`#`, `##`, `###`) for titles and subtitles.
2.  Format lists as Markdown lists (`-` or `1.`).
3.  Render all mathematical equations and symbols using LaTeX syntax (e.g., `$\\alpha^2 + \\beta^2 = \\gamma^2$`).
4.  If the page contains graphs, charts, or complex diagrams, provide a detailed, descriptive summary of the visual element within a Markdown blockquote. Example: `> A scatter plot showing a positive correlation...`
5.  Keep the output clean and focused on the content. Do not add any conversational text.
"""

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PDF_PATH = str(PROJECT_ROOT / "data" / "math55a.pdf")
OUTPUT_FILE_PATH = str(PROJECT_ROOT / "data" / "math55a.json")

MISSING_PAGE_PLACEHOLDER = "--- CONTENT MISSING: This page could not be processed owing to a persistent error. ---"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def _client():
    if not GOOGLE_API_KEY:
        raise ValueError("Please set your GEMINI_API_KEY in the .env file")
    return genai.Client(api_key=GOOGLE_API_KEY)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(ServerError),
)
async def async_call_llm(
    page: pymupdf.Page, page_num: int, client: genai.Client
) -> Dict[str, Any]:
    pix = page.get_pixmap(dpi=200)
    page_image = pix.tobytes("png")

    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=page_image, mime_type="image/png")
                ),
                types.Part(text=GEMINI_PROMPT),
            ]
        ),
    )
    return {"page_content": response.text, "page_num": page_num}


async def extract_content(
    pdf_path: str, concurrent_requests: int = 10
) -> List[Dict[str, Any]]:

    filepath = pathlib.Path(pdf_path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found at '{pdf_path}'")

    client = _client()

    semaphore = asyncio.Semaphore(concurrent_requests)

    async def async_extract_page_content(page, i, client):
        async with semaphore:
            try:
                return await async_call_llm(page, i, client)
            except Exception as e:
                return {
                    "page_num": i,
                    "page_content": MISSING_PAGE_PLACEHOLDER,
                    "error": f"Error processing page {i}: {e}",
                }

    with pymupdf.open(pdf_path) as pdf_document:
        tasks = [
            async_extract_page_content(page, i, client)
            for i, page in enumerate(pdf_document, start=1)
        ]
        results = await asyncio.gather(*tasks)
    return results


def store_pages_in_json(results, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_file_path


if __name__ == "__main__":
    results = asyncio.run(extract_content(PDF_PATH))
    store_pages_in_json(results, OUTPUT_FILE_PATH)
