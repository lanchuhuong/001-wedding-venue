import os
import sys

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

if os.getenv("OPENAI_API_KEY") is not None:
    from function.llm import get_llm_response
    from function.retriever import (
        check_existing_embeddings,
        initialize_retriever,
        query_documents,
        update_retriever,
    )

# Set page config
st.set_page_config(page_title="Chat Document", page_icon="🔍", layout="wide")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "supporting_info_sections" not in st.session_state:
    st.session_state.supporting_info_sections = []


# Center the title using Markdown and CSS
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
    </style>
    <div class="title"> Chat Document 👰 💐</div>
""",
    unsafe_allow_html=True,
)

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    if (api_key := os.getenv("OPENAI_API_KEY")) is None:
        api_key = st.text_input("Enter OpenAI API Key", type="password")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


def initialize_app():
    st.session_state.initialized = True
    load_dotenv(override=True)
    sys.path.append("..")
    with st.spinner("Initializing retriever..."):
        # Configure FAISS with persistence
        PERSIST_DIRECTORY = os.path.join(os.getcwd(), os.getenv("DATABASE_DIR"))
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        try:
            retriever = initialize_retriever()
            update_retriever(retriever)
            st.session_state.retriever = retriever

            # Check existing embeddings
            check_existing_embeddings(st.session_state.retriever.vectorstore)

            st.success("Retriever initialized successfully!")
        except Exception as e:
            raise e


def display_supporting_info(results):
    """Helper function to display supporting information."""
    # Create a container for all results
    with st.container():
        for doc_id, content in results.items():
            # Create a separate container for each document
            with st.container():
                st.subheader(f"Relevant images from: {content['company']}")
                st.write("🖼️ Images:")
                # Create columns for images if there are multiple
                if content["images"]:
                    columns = st.columns(
                        min(len(content["images"]), 3)
                    )  # Max 3 columns
                    for idx, image_doc in enumerate(content["images"]):
                        image_path = image_doc.metadata.get("image_path")
                        description = image_doc.page_content
                        if image_path and os.path.exists(image_path):
                            try:
                                with columns[
                                    idx % 3
                                ]:  # Use modulo to cycle through columns
                                    image = Image.open(image_path)
                                    st.image(
                                        image,
                                        caption=description,
                                        use_container_width=False,
                                    )
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")


# Main app layout
st.write("Ask questions about weddings and venues!")

# Initialize button
# if os.environ["OPENAI_API_KEY"] and st.session_state.initialized is False:
if api_key and st.session_state.initialized is False:
    initialize_app()

st.title("💬 Chatbot")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
# Display chat history with supporting information
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])
    # Display supporting information right after the assistant's message
    if msg["role"] == "assistant" and "supporting_docs" in msg:
        with st.expander("Supporting Information"):
            display_supporting_info(msg["supporting_docs"])


# Query input using chat input
if query := st.chat_input("Ask about wedding venues..."):
    if not st.session_state.retriever:
        st.error("Please initialize the system first!")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        with st.spinner("Searching..."):
            try:
                # Get retrieved documents
                results = query_documents(st.session_state.retriever, query)

                # Get LLM response
                llm_response = get_llm_response(
                    query, results, st.session_state.chat_history
                )

                response = st.chat_message("assistant").write_stream(llm_response)

                response = response.encode("utf-8").decode("utf-8")

                # Add assistant response to chat history with supporting docs
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "supporting_docs": results,
                    }
                )
                # Display supporting information for the current response
                with st.expander("Supporting Information"):
                    display_supporting_info(results)
            except Exception as e:
                st.error(f"Error: {e}")
