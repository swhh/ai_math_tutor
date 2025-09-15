# app.py (Stateful Paginator Version)
import os
from pathlib import Path
import uuid

import fitz
from langchain.schema import HumanMessage, AIMessage
import streamlit as st

from ai_math_tutor.backend import end_to_end_pipeline
from ai_math_tutor.config import PROJECT_ROOT

# --- Page Configuration ---
st.set_page_config(page_title="AI Mathematics Tutor", page_icon="📖", layout="wide")

# --- State Management ---
if "page_num" not in st.session_state:
    st.session_state.page_num = 1
if "rag_app" not in st.session_state:
    st.session_state.rag_app = None
if "page_chats" not in st.session_state:
    st.session_state.page_chats = {}
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "doc" not in st.session_state:
    st.session_state.doc = None  # Cache the opened document object
if "session_id" not in st.session_state:
    st.session_state.session_id = None


# --- Helper Functions for Page Navigation ---
def next_page():
    if st.session_state.page_num < st.session_state.doc.page_count:
        st.session_state.page_num += 1


def prev_page():
    if st.session_state.page_num > 1:
        st.session_state.page_num -= 1


def go_to_page(page_number):
    if 1 <= page_number <= st.session_state.doc.page_count:
        st.session_state.page_num = page_number


# --- Main App UI ---
st.title("📖 AI Mathematics Tutor")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Your Textbook")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process Textbook"):
            with st.spinner("Processing..."):
                original_name = Path(uploaded_file.name).name
                stem = Path(original_name).stem
                suffix = Path(original_name).suffix.lower()
                unique_name = f"{stem}__{uuid.uuid4().hex}{suffix}"
                file_path = os.path.join(str(PROJECT_ROOT), "data", unique_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                rag_app = end_to_end_pipeline(file_path)

                # Update session state after processing
                st.session_state.rag_app = rag_app
                st.session_state.processed_file = file_path
                st.session_state.doc = fitz.open(file_path)  # Cache the doc object
                st.session_state.page_chats = {}
                st.session_state.page_num = 1  # Reset to first page
                st.session_state.session_id = (
                    f"{unique_name}:{uuid.uuid4().hex}"  # start new session
                )
            st.success("Textbook processed successfully!")

# ---  Main Content Area ---
if st.session_state.processed_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Textbook Viewer")

        # --- NEW: Paginator Controls ---
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            st.button("⬅️ Previous", on_click=prev_page, use_container_width=True)
        with nav_col3:
            st.button("Next ➡️", on_click=next_page, use_container_width=True)
        with nav_col2:
            page_input = st.number_input(
                f"Page (1-{st.session_state.doc.page_count})",
                min_value=1,
                max_value=st.session_state.doc.page_count,
                value=st.session_state.page_num,
                key="page_input_key",
                on_change=lambda: go_to_page(st.session_state.page_input_key),
            )

        # Display the current page image
        page = st.session_state.doc.load_page(st.session_state.page_num - 1)
        pix = page.get_pixmap(dpi=150)
        st.image(pix.tobytes("png"), use_container_width=True)

    with col2:
        st.header("AI Mathematics Tutor")
        # Display chat messages
        current_page_history = st.session_state.page_chats.get(
            st.session_state.page_num, []
        )
        for message in current_page_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

        if prompt := st.chat_input(
            f"Ask a question about page {st.session_state.page_num}..."
        ):
            st.session_state.page_chats.setdefault(st.session_state.page_num, [])

            st.session_state.page_chats[st.session_state.page_num].append(
                HumanMessage(content=prompt)
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                inputs = {
                    "question": prompt,
                    "current_page_num": st.session_state.page_num,
                    "chat_history": current_page_history,
                    "collection_name": "calculus_textbook.json",  # modify this in future
                }

                response = ""
                compiled_workflow = st.session_state.rag_app.get_compiled_workflow()
                thread_id = (
                    f"{st.session_state.session_id}:p{st.session_state.page_num}"
                )
                for output in compiled_workflow.stream(
                    inputs,
                    {"recursion_limit": 5, "configurable": {"thread_id": thread_id}},
                ):
                    if "generate" in output:
                        response += output["generate"]["generation"]
                        message_placeholder.markdown(response + "▌")
                message_placeholder.markdown(response)

            st.session_state.page_chats[st.session_state.page_num].append(
                AIMessage(content=response)
            )
else:
    st.info("Please upload and process a PDF textbook in the sidebar to begin.")
