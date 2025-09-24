import asyncio
import json
import logging
import sqlite3
from typing import List, Optional, Union, TypedDict

from google.genai.errors import ClientError, ServerError
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph, END
from pydantic import BaseModel, Field

from ai_math_tutor.chunk_and_embed import (
    create_or_update_content_database,
    DEVICE,
    load_documents_from_json,
    chunk_and_embed_pipeline,
    extract_index,
    store_index,
)
from ai_math_tutor.config import (
    PROJECT_ROOT,
    SQLITE_DB_PATH,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    STRUCTURED_AI_ANSWERS,
)
from ai_math_tutor.extract_content_from_pdf import (
    extract_content,
    MISSING_PAGE_PLACEHOLDER,
    store_pages_in_json,
)
from ai_math_tutor.utils import (
    format_docs,
    get_filename_without_extension,
    parse_page_numbers,
)

GENERATION_LLM = "gemini-2.5-flash"
# use a smaller, weaker model for query analysis
QUERY_ANALYSIS_LLM = "gemini-1.5-flash"  


DEFAULT_K_CHUNKS = 8

GENERATE_PROMPT_TEMPLATE = """
        You are an expert mathematics tutor. Your goal is to help a user understand the textbook {collection_name}.

        Here is the recent conversation history:
        ---
        CHAT_HISTORY:
        {chat_history}
        ---

        The user is currently looking at the following content on page {current_page_num}:
        ---
        CURRENT_PAGE_CONTENT:
        {current_page_content}
        ---

        ---
        POTENTIALLY RELEVANT NEIGHBORING PAGE CONTENT:
        {neighboring_pages_content}
         ----
        Here is some related background context from other pages in the book:
        ---
        BACKGROUND_CONTEXT:
        {background_context}
        ---

        Based on all of the above, answer the user's latest question. If the question is a follow-up to the conversation, use the chat history to understand the context. Focus your answer on the CURRENT_PAGE_CONTENT and the user's immediate question.
        Keep answers brief at all times; reference page numbers (exclude the current page ({current_page_num}) from your page references) and cite key passages where you used information from the background context to formulate your response.

        LATEST QUESTION: {question}
        ANSWER:
        """

FAILURE_PROMPT_TEMPLATE = """
            You are an expert mathematics tutor. You have encountered an issue.
            Inform the user that you were unable to process the content for the current page (page {current_page_num}) during the initial textbook upload.
            Politely ask them to copy the text from the page they are viewing and paste it into the chat so you can help them with their question.

            Here is the user's question, which you should refer to:
            QUESTION: {question}
            
            Your response should be a polite request for the page content.
            """


QUERY_ANALYSIS_TEMPLATE = """You are an expert retrieval strategist for a RAG system.
            Your goal is to create a plan to fetch the best possible context for answering a user's question about their textbook.
            The user is currently on page {current_page_num}.
            ---
            CURRENT_PAGE_CONTENT:
            {current_page_content}

            The user's question is: {question}

            Based on their question, you will create a plan by calling the `RetrievalPlan` tool.
            - For simple definitions ('what is X?'), a small `k` is sufficient.
            - For complex explanations, proofs, or summaries, a larger `k` is needed.
            - If the question refers to content at the very beginning or end of a page, it is crucial to fetch the `neighboring_pages` (e.g., the previous or next page). 
            Provide the page numbers for the pages (besides that of the current page) needed to answer the user query. 
            ONLY provide page numbers if specific pages are needed to help answer the user query. 
            - Provide a list of keywords that can be looked up in the book index to find relevant pages to help answer the user question.
            """

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


class AIAnswer(BaseModel):
    """A structured AI answer containing page references.
    Only used if structured answer param in config set to true"""

    answer: str = Field(
        ..., description="The final, natural language answer to be shown to the user."
    )
    page_references: List[int] = Field(
        default_factory=list,
        description="A list of unique page numbers cited in the answer. This should be an empty list if no specific pages are cited.",
    )


class State(TypedDict):
    question: str
    current_page_num: int
    collection_name: str
    documents: Optional[List[Document]]
    generation: Union[AIAnswer, str]
    current_page_content: Optional[str]
    chat_history: List[BaseMessage]
    top_k_chunks: int  # number of documents to return
    neighboring_pages: List[int]
    neighboring_pages_content: Optional[str]
    index_keywords: List[str]


class RetrievalPlan(BaseModel):
    """A plan for retrieving context to answer a user's question."""

    top_k_chunks: int = Field(
        ...,
        description="The optimal number of semantically similar chunks to retrieve (top_k_chunks). "
        "Use a small number (3-5) for simple definitions. "
        "Use a larger number (8-15) for complex proofs, summaries, or explanations.",
    )

    neighboring_pages: List[int] = Field(
        default_factory=list,
        description="A list of specific, adjacent page numbers to fetch. "
        "For example, if the user asks about an equation at the top of page 50, "
        "you should include [49] to get the context from the previous page provided page 50 is not the start of a new chapter.",
    )
    index_keywords: List[str] = Field(
        default_factory=list,
        description="A list of keywords that are relevant to the user question",
    )


class RagApplication:

    def __init__(
        self, collection_name, content_db_path, vector_store=None, has_index=False
    ) -> None:
        self.collection_name = collection_name
        self.book_id = collection_name
        self.has_index = has_index

        self.generate_llm = init_chat_model(
            GENERATION_LLM, model_provider="google_genai", temperature=0
        )
        self.query_analysis_llm = init_chat_model(
            QUERY_ANALYSIS_LLM, model_provider="google_genai", temperature=0
        )

        self.content_db_conn = sqlite3.connect(content_db_path, check_same_thread=False)
        # Validate the vector store before using it
        if vector_store is not None:
            vector_store = self._validate_vector_store(vector_store)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            logger.info(f"Creating new vector store")
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE}
            )
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embedding_model,
                collection_name=self.collection_name,
            )

        self.compiled_workflow = self._generate_workflow()

    def close(self):
        """Close database connection"""
        if hasattr(self, "content_db_conn"):
            self.content_db_conn.close()

    def _format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        """Formats the chat history into a readable string for the prompt."""
        if not chat_history:
            return "No previous conversation history."
        return "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'Tutor'}: {msg.content}"
            for msg in chat_history
        )

    def _validate_vector_store(self, vector_store):
        """Validate vector store"""
        try:
            doc_count = vector_store._collection.count()
            if doc_count == 0:
                logger.warning(
                    f"Warning: Collection '{self.collection_name}' is empty, creating new vector store..."
                )
                vector_store = None  # Force creation of new vector store
            else:
                logger.warning(
                    f"Collection '{self.collection_name}' validated with {doc_count} documents"
                )
        except Exception as e:
            logger.warning(
                f"Warning: Could not validate collection '{self.collection_name}': {e}"
            )
            vector_store = None  # Force creation of new vector store
        return vector_store

    def analyse_query(self, state: State):
        question = state["question"]
        current_page_num = state["current_page_num"]
        current_page_content = state["current_page_content"]

        query_analyser = self.query_analysis_llm.with_structured_output(RetrievalPlan)

        prompt = ChatPromptTemplate.from_template(QUERY_ANALYSIS_TEMPLATE)

        inputs = {
            "current_page_num": current_page_num,
            "question": question,
            "current_page_content": current_page_content,
        }

        chain = prompt | query_analyser.with_retry(
            retry_if_exception_type=(ClientError, ServerError),
            wait_exponential_jitter=True,
            stop_after_attempt=3,
        )
        try:
            retrieval_plan = chain.invoke(inputs)
            neighboring_pages = sorted(
                list(set(retrieval_plan.neighboring_pages) - {current_page_num})
            )

            return {
                "top_k_chunks": retrieval_plan.top_k_chunks,
                "neighboring_pages": neighboring_pages,
                "index_keywords": retrieval_plan.index_keywords,
            }
        except Exception as e:  # if query analyser fails, set sensible defaults
            logger.error(
                f"Query analyser failure for {current_page_num} in book {self.book_id}: {e}"
            )

            neighboring_pages = [current_page_num - 1] if current_page_num > 1 else []
            return {
                "top_k_chunks": DEFAULT_K_CHUNKS,
                "neighboring_pages": neighboring_pages,
                "index_keywords": [],
            }

    def fetch_page(self, state: State):
        """Fetches the clean content for the current book and page from SQLite."""
        current_page_num = state["current_page_num"]
        cursor = self.content_db_conn.cursor()

        cursor.execute(
            "SELECT content FROM pages WHERE book_id = ? AND page_number = ?",
            (self.book_id, current_page_num),
        )
        result = cursor.fetchone()

        content = result[0] if result else "No content found for this page."
        return {"current_page_content": content}

    def fetch_neighboring_pages(self, state: State):
        """Fetch the markdown content for the neighboring pages from SQLite"""
        neighboring_pages_content = (
            "No relevant content available for neighboring pages."
        )

        neighboring_pages = state["neighboring_pages"]
        if neighboring_pages:
            cursor = self.content_db_conn.cursor()
            query = f"SELECT page_number, content FROM pages WHERE book_id = ? AND page_number IN ({','.join('?' for _ in neighboring_pages)})"
            params = [self.book_id] + neighboring_pages

            cursor.execute(query, params)
            results = cursor.fetchall()
            if results:
                results.sort(key=lambda x: x[0])
                neighboring_pages_content = "\n\n".join(
                    f"--- Content from Page {num} ---\n{text}" for num, text in results
                )

        return {"neighboring_pages_content": neighboring_pages_content}

    def retrieve(self, state: State):
        """Fetch relevant docs from vector db"""
        question = state["question"]
        current_page = state["current_page_num"]
        top_k_chunks = state["top_k_chunks"]

        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": top_k_chunks,
                "filter": {"page_num": {"$lte": current_page}},
            }
        )

        documents = retriever.invoke(question)
        # fallback: remove filter if no documents returned
        if not documents:
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": max(top_k_chunks, DEFAULT_K_CHUNKS)}
            )
            documents = retriever.invoke(question)
        return {
            "documents": documents,
            "question": question,
            "current_page_num": current_page,
        }

    def route_question(self, state: State):
        """Route workflow based on current state"""
        has_page = bool(state.get("current_page_content"))
        has_docs = bool(state.get("documents"))
        has_neighbors = bool(state.get("neighboring_pages_content"))
        if has_page and has_docs and has_neighbors:
            return "generate"
        if not has_page:
            return "fetch_page"
        return "analyse_query"

    def fetch_keyword_page_nums(self, state: State):
        """If there is an index and relevant keywords, return page numbers"""
        index_keywords = state["index_keywords"]
        neighboring_pages = state["neighboring_pages"]
        index_pages = None

        if not index_keywords:
            return {"neighboring_pages": neighboring_pages}

        try:
            cursor = self.content_db_conn.cursor()
            search_query = " OR ".join(f'"{kw}"' for kw in index_keywords)
            query = f"""
            SELECT pages_json FROM book_index_fts
            WHERE book_id = ? AND term MATCH ?
            ORDER BY rank
            LIMIT 5;
            """
            params = [self.book_id, search_query]

            cursor.execute(query, params)
            results = cursor.fetchall()

            if results:
                page_strings = [
                    page_str for res in results for page_str in json.loads(res[0])
                ]
                index_pages = parse_page_numbers(page_strings)

            if index_pages:
                neighboring_pages = list(set(neighboring_pages + index_pages))
        except Exception as e:
            logger.warning(f"Skipping index lookup due to error: {e}")  
        return {"neighboring_pages": neighboring_pages}

    def generate(self, state: State):
        """Generate LLM response to user query"""
        question = state["question"]
        current_page_content = state["current_page_content"]
        documents = state["documents"]
        current_page_num = state["current_page_num"]
        collection_name = state["collection_name"]
        chat_history = state["chat_history"]
        neighboring_pages_content = state["neighboring_pages_content"]

        # if page markdown generation failed
        if MISSING_PAGE_PLACEHOLDER in current_page_content:
            prompt_template = FAILURE_PROMPT_TEMPLATE
            inputs = {"question": question, "current_page_num": current_page_num}
        else:
            prompt_template = GENERATE_PROMPT_TEMPLATE
            background_context = format_docs(documents)
            inputs = {
                "current_page_content": current_page_content,
                "background_context": background_context,
                "question": question,
                "current_page_num": current_page_num,
                "collection_name": collection_name,
                "chat_history": self._format_chat_history(chat_history),
                "neighboring_pages_content": neighboring_pages_content,
            }

        prompt = ChatPromptTemplate.from_template(prompt_template)
        # if generate AI providing structured answers, use structured output
        if STRUCTURED_AI_ANSWERS:
            structured_llm = self.generate_llm.with_structured_output(AIAnswer)
            rag_chain = prompt | structured_llm.with_retry(
                retry_if_exception_type=(ClientError, ServerError),
                wait_exponential_jitter=True,
                stop_after_attempt=3,
            )
        else:
            rag_chain = (
                prompt
                | self.generate_llm.with_retry(
                    retry_if_exception_type=(ClientError, ServerError),
                    wait_exponential_jitter=True,
                    stop_after_attempt=3,
                )
                | StrOutputParser()
            )
        try:
            generation = rag_chain.invoke(inputs)
            return {"generation": generation}
        except Exception as e:
            logger.exception(
                f"Generate failure for page {current_page_num} in {self.book_id}: {e}"
            )
            if MISSING_PAGE_PLACEHOLDER in current_page_content:
                fallback = f"""Sorry I could not process page {current_page_num}. 
                    Please copy the text from the page and ask your question again.
                    Your question was: {question}"""
            else:
                fallback = f"""I'm having trouble generating a response right now. Please try again shortly. 
                If the issue persists, consider rephrasing your question for page {current_page_num}"""

            if STRUCTURED_AI_ANSWERS:
                return {"generation": AIAnswer(answer=fallback, page_references=[])}
            return {"generation": fallback}

    def _generate_workflow(self):
        """Generate workflow to be compiled"""
        workflow = StateGraph(State)

        workflow.add_node("fetch_page", self.fetch_page)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        workflow.add_node("analyse_query", self.analyse_query)
        if self.has_index:
            workflow.add_node(
                "fetch_keyword_page_nums", self.fetch_keyword_page_nums
            )  # only add node to workflow if index exists
        workflow.add_node("fetch_neighboring_pages", self.fetch_neighboring_pages)

        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "fetch_page": "fetch_page",
                "analyse_query": "analyse_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("fetch_page", "analyse_query")
        workflow.add_edge("analyse_query", "retrieve")

        # only add fetch_keyword_page_nums to workflow if an index exists
        if self.has_index:
            workflow.add_edge("analyse_query", "fetch_keyword_page_nums")
            workflow.add_edge("fetch_keyword_page_nums", "fetch_neighboring_pages")
        else:
            workflow.add_edge("analyse_query", "fetch_neighboring_pages")
        workflow.add_edge(["fetch_neighboring_pages", "retrieve"], "generate")  # join
        workflow.add_edge("generate", END)

        checkpointer = InMemorySaver()  # keep track of workflow state

        return workflow.compile(checkpointer=checkpointer)

    def get_compiled_workflow(self):
        return self.compiled_workflow


def end_to_end_pipeline(pdf_path):
    """Generate rag app instance from textbook pdf file path"""
    book_id = get_filename_without_extension(pdf_path)

    results = asyncio.run(extract_content(pdf_path))  # convert pdf pages to markdown

    json_output_file_path = str(PROJECT_ROOT / "data" / f"{book_id}.json")
    store_pages_in_json(results, json_output_file_path)  # store markdown pages in json
   
    # convert json data into Langchain Documents
    documents = load_documents_from_json(json_output_file_path)  

    try:
        create_or_update_content_database(documents, book_id)  # add pages to sqlite db
    except sqlite3.Error as e:
        logger.exception(f"Book upload to SQLite failed for '{book_id}': {e}")
        raise

    book_index = extract_index(documents)  # extract book index
    has_index = book_index.index_found
    if has_index:  # if book index extraction successful
        try:
            store_index(book_index, book_id, SQLITE_DB_PATH)  # store book index
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")
            has_index = False

    vector_store = None
    try:
        # chunk and embed documents
        vector_store = chunk_and_embed_pipeline(documents, collection_name=book_id)  
        rag_app = RagApplication(
            collection_name=book_id,
            content_db_path=SQLITE_DB_PATH,
            vector_store=vector_store,
            has_index=has_index,
        )  # create rag app instance

        return rag_app
    # if ragapp creation process fails, remove chromadb embeddings and metadata
    except Exception as e:  
        try:
            if vector_store is not None:
                vector_store._client.delete_collection(name=book_id)
        except Exception as cleanup_error:
            logger.warning(
                f"Failed to clean up ChromaDB collection {book_id}: {cleanup_error}"
            )
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="AI Mathematics Tutor",
        description="Generates a rag app to read a mathematics textbook or paper with the reader",
    )
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename
    pdf_path = str(PROJECT_ROOT / "data" / filename)

    collection_name = get_filename_without_extension(pdf_path)

    rag_app = end_to_end_pipeline(pdf_path)
    compiled_workflow = rag_app.get_compiled_workflow()

    chat_history = []

    print(
        f"\n--- Studying '{collection_name}'. Type 'exit' or 'quit' to end. Type '/new' to start a new conversation ---"
    )

    session_id = f"{collection_name}"
    import uuid

    while True:
        try:
            user_question = input(f"\nAsk a question about {collection_name}: ")
            if user_question.lower() in ["exit", "quit"]:
                break

            if user_question.lower() == "/new":
                chat_history = []
                session_id = f"{session_id}:{uuid.uuid4().hex}"  # start new session; effectively workflow state graph cache
                print("--- New conversation started. ---")
                continue
            current_page = int(input("What page are you on? "))
            if current_page <= 0:
                raise ValueError

            inputs = {
                "question": user_question,
                "current_page_num": current_page,
                "collection_name": collection_name,
                "chat_history": chat_history,
            }

            thread_id = f"{session_id}:p{current_page}"
            for output in compiled_workflow.stream(
                inputs, {"configurable": {"thread_id": thread_id}}
            ):
                for key, value in output.items():
                    if key == "retrieve" and "documents" in value:
                        print(f'top_k_chunks: {len(value["documents"])}')
                    print(f"Finished node '{key}':")

            final_generation = value["generation"]
            print("\nAI Tutor:", final_generation)

            chat_history.append(HumanMessage(content=user_question))
            chat_history.append(AIMessage(content=final_generation))

        except (ValueError, TypeError):
            print("Invalid page number...")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            rag_app.close()  # close SQLite connection


if __name__ == "__main__":
    main()
