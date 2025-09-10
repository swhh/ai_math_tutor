import asyncio
import sqlite3
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.document import Document
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from ai_math_tutor.chunk_and_embed import CHROMA_DB_PATH, create_or_update_content_database, DEVICE, EMBEDDING_MODEL_NAME, load_documents_from_json, chunk_and_embed_pipeline, SQLITE_DB_PATH
from ai_math_tutor.extract_content_from_pdf import extract_content, MISSING_PAGE_PLACEHOLDER, OUTPUT_FILE_PATH, store_pages_in_json
from ai_math_tutor.utils import format_docs, get_filename_without_extension

GENERATION_LLM = "gemini-2.5-flash"
QUERY_ANALYSIS_LLM = "gemini-1.5-flash" # use a smaller, weaker model for grading
DEFAULT_DOC_NUM = 12
SAMPLE_BOOK = 'calculus_textbook.json'
CONVERSATION_LIMIT = 10

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

        Here is some related background context from other pages in the book:
        ---
        BACKGROUND_CONTEXT:
        {background_context}
        ---

        Based on all of the above, answer the user's latest question. If the question is a follow-up to the conversation, use the chat history to understand the context. Focus your answer on the CURRENT_PAGE_CONTENT and the user's immediate question.
        Keep answers brief at all times and reference page numbers where you used information from the background context to formulate your response.

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


class State(TypedDict):
    question: str
    current_page_num: int
    collection_name: str
    documents: List[Document]
    generation: str
    current_page_content: str
    chat_history: List[BaseMessage]


class RagApplication:

    def __init__(self, collection_name, content_db_path) -> None:
        self.collection_name = collection_name
        self.book_id = collection_name

        self.generate_llm = init_chat_model(GENERATION_LLM, model_provider="google_genai", temperature=0)
        
        self.content_db_conn = sqlite3.connect(content_db_path, check_same_thread=False)

        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': DEVICE})

        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model,
            collection_name=self.collection_name
        )

        self.compiled_workflow = self._generate_workflow()

    def _format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        """Formats the chat history into a readable string for the prompt."""
        if not chat_history:
            return "No previous conversation history."
        return "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'Tutor'}: {msg.content}"
            for msg in chat_history
        )

    def fetch_page(self, state: State):
        """Fetches the clean content for the current book and page from SQLite."""
        current_page_num = state["current_page_num"]
        cursor = self.content_db_conn.cursor()
        
        cursor.execute(
            "SELECT content FROM pages WHERE book_id = ? AND page_number = ?",
            (self.book_id, current_page_num)
        )
        result = cursor.fetchone()
    
        content = result[0] if result else "No content found for this page."
        return {"current_page_content": content}
    

    def retrieve(self, state: State):
        """Fetch relevant docs from vector db"""
        question = state["question"]
        current_page = state["current_page_num"]
        
        retriever = self.vector_store.as_retriever(
            search_kwargs={'k': DEFAULT_DOC_NUM, 
                           'filter': {'page_number': {'$lte': current_page}}}
        )
        
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question, "current_page_num": current_page}

    def generate(self, state: State):
        """Generate LLM response to user query"""
        question = state["question"]
        current_page_content = state["current_page_content"]
        documents = state["documents"]
        current_page_num = state["current_page_num"]
        collection_name = state["collection_name"]
        chat_history = state['chat_history']

        if MISSING_PAGE_PLACEHOLDER in current_page_content:  # if page markdown generation failed
            prompt_template = FAILURE_PROMPT_TEMPLATE
            inputs = {"question": question, 
                    "current_page_num": current_page_num}
        else:
            prompt_template = GENERATE_PROMPT_TEMPLATE
            background_context = format_docs(documents)
            inputs = {
            "current_page_content": current_page_content,
            "background_context": background_context,
            "question": question,
            "current_page_num": current_page_num,
            "collection_name": collection_name,
            "chat_history": self._format_chat_history(chat_history)
            }

        prompt = ChatPromptTemplate.from_template(prompt_template)

        rag_chain = prompt | self.generate_llm | StrOutputParser()

        generation = rag_chain.invoke(inputs)
        return {"generation": generation}

    def _generate_workflow(self):
        """Generate workflow to be compiled"""
        workflow = StateGraph(State)

        workflow.add_node('fetch_page', self.fetch_page)
        workflow.add_node('retrieve', self.retrieve)
        workflow.add_node('generate', self.generate)

        workflow.set_entry_point('fetch_page')
        workflow.add_edge('fetch_page', 'generate')

        workflow.set_entry_point('retrieve')
        workflow.add_edge('retrieve', 'generate')
        workflow.add_edge('generate', END)

        return workflow.compile()

    def get_compiled_workflow(self):
        return self.compiled_workflow


def end_to_end_pipeline(pdf_path):
    """Generate rag app instance from textbook pdf file path"""
    book_id = get_filename_without_extension(pdf_path) 

    results = asyncio.run(extract_content(pdf_path)) # convert pdf pages to markdown
    output_file_path = store_pages_in_json(results, OUTPUT_FILE_PATH) # store markdown pages in json
    documents = load_documents_from_json(output_file_path) # convert json data into Langchain docs
    create_or_update_content_database(documents, book_id) # add pages to sqlite db
    
    chunk_and_embed_pipeline(documents, collection_name=book_id) # chunk and embed documents

    rag_app = RagApplication(collection_name=book_id, content_db_path=SQLITE_DB_PATH) # create rag app instance

    return rag_app


def main():

    load_dotenv()

    collection_name = SAMPLE_BOOK

    rag_app = RagApplication(collection_name=collection_name, content_db_path=SQLITE_DB_PATH)
    compiled_workflow = rag_app.get_compiled_workflow()

    chat_history = []

    print(f"\n--- Studying '{collection_name}'. Type 'exit' or 'quit' to end. Type '/new' to start a new conversation ---")

    while True:
            try:
                user_question = input(f"\nAsk a question about {collection_name}: ")
                if user_question.lower() in ["exit", "quit"]:
                    break
                
                if user_question.lower() == "/new" or len(chat_history) == CONVERSATION_LIMIT:
                    chat_history = []
                    print("--- New conversation started. ---")
                    continue

                current_page = int(input("What page are you on? "))
                if current_page <= 0:
                    raise ValueError

                inputs = {
                    "question": user_question,
                    "current_page_num": current_page,
                    "collection_name": collection_name,
                    "chat_history": chat_history 
                }
                
                for output in compiled_workflow.stream(inputs, {"recursion_limit": 5}):
                    for key, value in output.items():
                        print(f"Finished node '{key}':")
                
                final_generation = value['generation']
                print("\nAI Tutor:", final_generation)

                chat_history.append(HumanMessage(content=user_question))
                chat_history.append(AIMessage(content=final_generation))

            except (ValueError, TypeError):
                print("Invalid page number...")
            except Exception as e:
                print(f"\nAn error occurred: {e}")


if __name__ == '__main__':
    main()


    
