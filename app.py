import os
import sys

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from function.llm import get_llm_response
from function.retriever import (
    check_existing_embeddings,
    initialize_retriever,
    query_documents,
    update_retriever,
)

load_dotenv(override=True)
sys.path.append("..")

# Set page config
st.set_page_config(page_title="Chat Document", page_icon="🔍", layout="wide")

chat_history = []
# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Configure FAISS with persistence
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "faiss_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)


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
    if (api_key := os.environ.get("OPENAI_API_KEY")) is None:
        api_key = st.text_input("Enter OpenAI API Key", type="password")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.write(os.getenv("OPENAI_API_KEY"))


# In your Streamlit initialization
def initialize_app():
    with st.spinner("Initializing retriever..."):
        try:
            retriever = initialize_retriever()
            update_retriever(retriever)
            st.session_state.retriever = retriever

            # Check existing embeddings
            check_existing_embeddings(st.session_state.retriever.vectorstore)

            st.success("Retriever initialized successfully!")
        except Exception as e:
            raise e


# Main app layout
st.write("Ask questions about weddings and venues!")
st.write(PERSIST_DIRECTORY)

# Initialize button
if api_key and st.session_state.retriever is None:
    initialize_app()

st.title("💬 Chatbot")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
# Display user message
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])


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

                # Display assistant response
                # st.chat_message("assistant").write(llm_response)

                # Display supporting information in an expander
                with st.expander("View Supporting Information"):
                    for doc_id, content in results.items():
                        st.subheader(f"Results from: {content['company']}")
                        st.write("🖼️ Images:")
                        for image_doc in content["images"]:
                            image_path = image_doc.metadata.get("image_path")
                            description = image_doc.page_content
                            # st.write(f"Description: {description}")
                            if image_path and os.path.exists(image_path):
                                try:
                                    image = Image.open(image_path)
                                    st.image(
                                        image,
                                        caption=description,
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.error(f"Error displaying image: {e}")

            except Exception as e:
                st.error(f"Error processing query: {e}")

# Add a reset button in the sidebar to clear the database and chat history
# with st.sidebar:
#     if st.button("Reset Database"):
#         try:
#             if st.session_state.chroma_client:
#                 st.session_state.chroma_client.reset()
#             if os.path.exists(PERSIST_DIRECTORY):
#                 import shutil

#                 shutil.rmtree(PERSIST_DIRECTORY)
#             st.session_state.retriever = None
#             st.session_state.chroma_client = None
#             st.session_state.chat_history = []
#             st.success("Database and chat history reset successfully!")
#             st.experimental_rerun()
#         except Exception as e:
#             st.error(f"Error resetting database: {e}")
