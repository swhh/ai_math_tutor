import json
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from typing import List, Dict


