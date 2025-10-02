from typing import List, Literal

from langchain.schema import ChatMessage
from pydantic import BaseModel
from sqlmodel import SQLModel

class Token(BaseModel):
    access_token: str
    token_type: str

class UserRead(SQLModel):
    """The response model for returning user information, avoiding returning hashed password"""
    id: int
    username: str

class UserCreate(SQLModel):
    """The model for receiving user registration data."""
    username: str
    password: str

class BookRead(SQLModel):
    id: int
    title: str
    ingestion_status: str

class ChatMessage(BaseModel):
    role: Literal["user", "tutor"]
    content: str

# --- 2. The main response model for the chat history ---
class ChatHistoryRead(BaseModel):
    page_number: int
    messages: List[ChatMessage]

class QueryRequest(BaseModel):
    """The request model for asking a question."""
    question: str

