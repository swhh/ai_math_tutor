# AI Math Tutor

## Introduction

A RAG-based AI reading companion for mathematics textbooks and papers. A reader can query the AI reader where he/she has questions about a passage, concept, definition or proof. The AI reader will review the current page along with any relevant content from neighboring pages, semantically similar passages elsewhere in the book and pages in the index which are related to the user question and then provide an answer with page references and citations where appropriate.


## How It Works
The LangChain-based Python app ingests a PDF, extracts page-level content and converts it to markdown with Gemini, builds embeddings for retrieval, and answers questions grounded in the current page, neighbouring pages, relevant pages fetched from the index (if available) via a FTS in SQLite and semantically relevant chunks at pages earlier in the book (i.e. pages that the student will have already read to get to the current page) whose embeddings are stored in a vector database. 

## Features
- **End-to-end pipeline**: PDF → page markdown JSON → SQLite content DB → chunk+embed → interactive tutoring.
- **Grounded answers**: Uses current page, neighboring pages, relevant indexed pages and retrieved background chunks.
- **Adaptive retrieval**: Query analysis selects `k` chunks, neighboring pages, and index keywords.
- **Book index support**: Parses a detected index and stores it in SQLite with FTS5 for keyword lookups if there is an index.
- **Local vector store**: ChromaDB with HuggingFace embeddings.

## Architecture
- **Extraction**: Converts PDF pages to markdown using Gemini; stores in `data/<book>.json` and `data/book_content.db`.
- **Indexing**: Optionally parses the last ~10% (up to a lookback limit) to detect and store the book index in SQLite.
- **Embedding**: Markdown chunks get embedded into a Chroma collection for each book.
- **RAG workflow**: Orchestrated with LangGraph; routes between page fetch, query analysis, neighbouring page fetches, retrieval, and generation; leverages langraph thread caching to avoid repeating workflow steps for follow-up questions.

## Roadmap 
- Tablet Reading app with passage highlighting.
- Citations with page links in AI maths tutor responses.
- Capability to have the AI tutor test the student at end of chapters based on chapter material and on dialogues within the chapter which indicate student strengths and weaknesses.
- Export conversations and study notes.
