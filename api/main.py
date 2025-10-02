import asyncio
from datetime import timedelta
import functools
import logging
from pathlib import Path
import sys
import threading
from typing import Annotated, List

from cachetools import LRUCache
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response, status, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
import fitz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import AIMessage, Document, HumanMessage, messages_from_dict, messages_to_dict
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

from ai_math_tutor.security import (create_access_token,
                    get_current_user,
                    get_password_hash,
                    verify_password, 
                    ACCESS_TOKEN_EXPIRE_MINUTES,
                    get_session)
from ai_math_tutor.database.session import get_session, get_session_context
from ai_math_tutor.database.models import Book, Bookmark, ChatHistory, IngestionStatus, PageContent, User
from ai_math_tutor.utils import generate_collection_name, save_upload_file_async
from ai_math_tutor.chunk_and_embed import chunk_and_embed_pipeline, DEVICE, load_documents_from_json
from ai_math_tutor.extract_content_from_pdf import extract_content, store_pages_in_json
from ai_math_tutor.config import  EMBEDDING_MODEL_NAME
from ai_math_tutor.backend import ProductionRagApplication
from ai_math_tutor.embeddings import get_embedding_model

from .schemas import (
    BookRead,
    ChatHistoryRead,
    ChatMessage,
    QueryRequest,
    Token,
    UserCreate,
    UserRead,
)

# start cache
RAG_APP_CACHE = LRUCache(maxsize=100) 

# Lock to prevent race conditions for FastAPI threading
CACHE_LOCK = threading.Lock()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


def get_rag_app(collection_name: str) -> ProductionRagApplication:
    """Get a RAG application instance from the thread-safe closable LRUCache"""
    if collection_name in RAG_APP_CACHE:
        logger.info(f"CACHE HIT: Reusing existing RAG app instance for collection: {collection_name}")
        return RAG_APP_CACHE[collection_name]

    with CACHE_LOCK:
        if collection_name in RAG_APP_CACHE:
            return RAG_APP_CACHE[collection_name]
        
        logger.info(f"CACHE MISS: Creating new RAG app instance for collection: {collection_name}")
        shared_embedding_model = get_embedding_model()
        instance = ProductionRagApplication(collection_name=collection_name, embedding_model=shared_embedding_model)
        RAG_APP_CACHE[collection_name] = instance
        return instance


def store_pages_in_postgres(documents: List[Document], book, session):
    for document in documents:
        page_content = PageContent(book_id=book.id,
                                    page_number=document.metadata['page_num'],
                                    content=document.page_content)
        session.add(page_content)


def ingestion_pipeline(book_id: int, pdf_path):
    vector_store = None

    with get_session_context() as session:
        try:
            book = session.get(Book, book_id)
            book.ingestion_status = IngestionStatus.PROCESSING_CONTENT
            session.commit()

            results = asyncio.run(extract_content(pdf_path))  # convert pdf pages to markdown

            json_output_file_path = str(DATA_DIR / f"{book.collection_name}.json")
            store_pages_in_json(results, json_output_file_path)  # store markdown pages in json
        
            # convert json data into Langchain Documents
            documents = load_documents_from_json(json_output_file_path) 

            store_pages_in_postgres(documents, book, session) # store markdown in postgres

            # chunk and embed
            vector_store = chunk_and_embed_pipeline(documents, collection_name=book.collection_name) 

            book.ingestion_status = IngestionStatus.COMPLETE
            session.commit()

        except Exception:
            if book:
                book.ingestion_status = IngestionStatus.FAILED
                session.commit()
            if vector_store:
                try:
                    vector_store._client.delete_collection(name=book.collection_name)
                except Exception as cleanup_error:
                    logger.exception(f"Failed to clean up vector store for book {book.id}: {cleanup_error}")


# --- Initialize the FastAPI app ---
app = FastAPI(
    title="AI Math Tutor API",
    description="An API for interacting with a conversational maths textbook tutor.",
    version="0.1.0",
)

SessionDep = Annotated[Session, Depends(get_session)]

# --- Define API Endpoints ---

@app.post("auth/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDep):
    """Authentication endpoint for generating JWTs"""
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, get_password_hash(form_data.password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

# --- A Protected Endpoint for Testing ---
@app.get("/users/me", response_model=UserRead, tags=["Users"])
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    """Get current user's profile"""
    return current_user


# --- A temporary endpoint to create a user for testing ---
@app.post("/users/create", response_model=UserRead, tags=["Users"])
def create_user(username: str, password: str, session: Session = Depends(get_session)):
    """A simple endpoint to create a user for testing purposes."""
    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    return new_user

@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the AI Math Tutor API!"}


@app.post("/books/upload", tags=["General"], response_model=BookRead, status_code=201)
async def upload_pdf(pdf_file: UploadFile, current_user: Annotated[User, Depends(get_current_user)], 
                    background_tasks: BackgroundTasks, session: SessionDep):
    """Upload pdf book"""
    title = Path(Path(pdf_file.filename).name).stem
    collection_name = generate_collection_name(pdf_file.filename)
    destination = DATA_DIR / collection_name

    await save_upload_file_async(pdf_file, destination)

    try:
        new_book = Book(user_id=current_user.id,
                        title=title,
                        s3_key=str(destination),
                        collection_name=collection_name,
                        ingestion_status=IngestionStatus.UPLOADED)
        session.add(new_book)

        new_bookmark = Bookmark(
            user=current_user, 
            book=new_book,     
            page_number=1
        )
        session.add(new_bookmark) 
        session.commit()
        session.refresh(new_book)
        session.refresh(new_bookmark)
       
    except Exception as e:
        session.rollback()
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=500, detail=f"Database transaction failed: {e}")

    # run ingestion pipeline in background
    background_tasks.add_task(ingestion_pipeline, new_book.id, destination)

    return new_book    


@app.post("/users/register", response_model=UserRead, status_code=status.HTTP_201_CREATED, tags=["Users"])
def register_user(user_data: UserCreate, session: SessionDep):
    """Register user"""
    hashed_password = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_password)
    try:
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail="Username already exists. Please choose a different one."
        )
    return new_user


@app.get("/books")
def get_books(current_user: Annotated[User, Depends(get_current_user)], session: SessionDep):
    """Get details of all user books"""
    books = session.exec(select(Book).where(Book.user_id == current_user.id)).all()
    if not books:
        raise HTTPException(status_code=404, detail="No books found")    
    return books


@app.get("books/{book_id}", response_model=BookRead, tags=["RAG"])
def get_book(book_id: int, current_user: Annotated[User, Depends(get_current_user)], session: SessionDep):
    """Get details of book book_id"""
    book = session.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail=f"No book with id {book_id} found")   
    return book


@app.get("/books/{book_id}/chat/{page_number}", response_model=ChatHistoryRead, tags=["Chat"])
def get_page_chat_history(book_id: int, page_number: int, current_user: Annotated[User, Depends(get_current_user)], session: SessionDep):
    """Get chat history for page_number in book book_id"""
    history = session.exec(
                select(ChatHistory).where(
                    ChatHistory.book_id == book_id,
                    ChatHistory.page_number == page_number,
                    ChatHistory.user_id == current_user.id 
                )
            ).first()
    if not history:
        return ChatHistoryRead(page_number=page_number, messages=[])

    processed_messages = []

    for message in messages_from_dict(history.messages):
        content_text = message.content
        role = 'user' if isinstance(message, HumanMessage) else 'tutor'
        processed_messages.append(ChatMessage(role=role, content=content_text))
        
    return ChatHistoryRead(page_number=history.page_number, messages=processed_messages)
    

@app.post("/books/{book_id}/chat/{page_number}", tags=["Chat"])
async def ask_question(
                book_id: int, 
                page_number: int, 
                query_request: QueryRequest, 
                current_user: Annotated[User, Depends(get_current_user)], 
                session: SessionDep):
    """Ask the AI a question for page. NB: assumes structured content"""
    prompt = query_request.question
    if not prompt:
        raise HTTPException(status_code=404, detail="No question asked")
    # get book
    book = session.get(Book, book_id)
    if not book or book.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Book not found or access denied.")

    # get chat history
    history = session.exec(
                select(ChatHistory).where(
                    ChatHistory.book_id == book_id,
                    ChatHistory.page_number == page_number,
                    ChatHistory.user_id == current_user.id 
                )
            ).first()

    if history and history.messages:
        chat_history = messages_from_dict(history.messages)
    else:
        chat_history = []

    # get rag app
    rag_app = get_rag_app(book.collection_name)
    # 
    inputs = {
                    "question": prompt,
                    "current_page_num": page_number,
                    "chat_history": chat_history,
                    "collection_name": book.collection_name,
                    "book_id": book_id
                }
    compiled_workflow = rag_app.get_compiled_workflow()
                
    config = {"configurable": {"db_session": session}}
    # generate AI response; NB: currently only using structured output so no response streaming
    try:
        final_state = await compiled_workflow.ainvoke(inputs, config)
        generation = final_state.get('generation')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG processing: {e}")

    if not generation:
        HTTPException(status_code=500, detail=f"Failed to generate a response")

    # update chat history
    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=generation.answer, 
                            additional_kwargs={"structured_answer": generation.dict()}))
    serialised_chat_history = messages_to_dict(chat_history)

    if history:
        history.messages = serialised_chat_history
    else:
        history = ChatHistory(user_id=current_user.id,
                              book_id=book_id,
                                page_number=page_number,
                                messages=serialised_chat_history)
    try:
        session.add(history)
        session.commit()
    except Exception as e:
        session.rollback()
        logger.exception(f"CRITICAL: Failed to save chat history for book {book_id}, page {page_number}. Error: {e}")

    return generation


@app.get("/books/{book_id}/pages/{page_number}/image", tags=["Pages"])
async def get_page_image(
    book_id: int,
    page_number: int,
    session: SessionDep,
    current_user: Annotated[User, Depends(get_current_user)]
):
    """Fetches a specific page of a book and returns it as a PNG image. Core endpoint for the PDF reader UI"""
    book = session.get(Book, book_id)
    if not book or book.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Book not found or access denied.")
   
    pdf_path = book.s3_key
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_number - 1) 
            # Render at a good resolution for web/tablet display
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        # Handle cases where the PDF file might be missing or corrupt
        raise HTTPException(status_code=500, detail=f"Failed to render page: {e}")
