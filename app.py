import base64
import concurrent.futures
import io
import os
import sys
from functools import lru_cache
from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image

from function.llm import get_llm_response
from function.retriever import (
    check_existing_embeddings,
    initialize_retriever,
    load_venue_metadata,
    query_documents,
    update_retriever,
)

bucket_name = "wedding-venues-001"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)


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
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if "current_supporting_docs" not in st.session_state:
    st.session_state.current_supporting_docs = None

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
    st.title("Supporting Information:")
    if st.session_state.current_supporting_docs:
        display_supporting_info(st.session_state.current_supporting_docs)


# Cache for image data
@st.cache_data(ttl=3600, max_entries=100)  # Limit cache size
def get_cached_image_data(blob_path: str) -> bytes:
    blob = bucket.blob(blob_path)
    # Add image compression if size > 1MB
    image_data = blob.download_as_bytes()
    if len(image_data) > 1_000_000:  # 1MB
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail((800, 800))  # Resize large images
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
    return image_data


def preload_images(company_name: str, image_paths: List[str]) -> Dict[str, bytes]:
    """Preload multiple images concurrently"""

    def fetch_single_image(image_path: str) -> tuple[str, bytes]:
        image_filename = os.path.basename(image_path)
        full_path = f"processed/adobe_extracted/{company_name}/figures/{image_filename}"
        image_data = get_cached_image_data(full_path)
        return image_path, image_data

    # Use ThreadPoolExecutor for concurrent downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_single_image, path) for path in image_paths]

        # Create dictionary of successful image loads
        image_data_dict = {}
        for future in concurrent.futures.as_completed(futures):
            path, data = future.result()
            if data is not None:
                image_data_dict[path] = data

        return image_data_dict


def display_supporting_info(results, images_per_venue=3):
    """
    Display venue information with progressive image loading.

    Args:
        results: Dictionary of venue results
        images_per_venue: Number of images to show initially per venue
    """
    if not results:
        return

    for doc_id, content in results.items():
        st.sidebar.subheader(f" 💒 {content['company']}")

        # Display venue metadata
        metadata = next(iter(content["images"] + content["text"])).metadata

        with st.sidebar.container():
            st.markdown('<div class="venue-info">', unsafe_allow_html=True)

            if metadata.get("website"):
                st.markdown(f"🌐 [Website]({metadata['website']})")

            if metadata.get("phone") and pd.notna(metadata["phone"]):
                st.markdown(f"📞 {metadata['phone']}")

            if metadata.get("address"):
                st.markdown(f"📍 {metadata['address']}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Handle images with progressive loading
        if content["images"]:
            st.sidebar.write("🖼️ Venue Images:")

            # Get state key for this venue
            state_key = f"show_all_images_{content['company']}"
            if state_key not in st.session_state:
                st.session_state[state_key] = False

            # Determine how many images to show
            total_images = len(content["images"])
            if st.session_state[state_key]:
                images_to_show = content["images"]
            else:
                images_to_show = content["images"][:images_per_venue]

            # Collect image paths for the images we'll display
            image_paths = [
                doc.metadata.get("image_path")
                for doc in images_to_show
                if doc.metadata.get("image_path")
            ]

            # Preload selected images
            image_data_dict = preload_images(content["company"], image_paths)

            # Display images
            for image_doc in images_to_show:
                image_path = image_doc.metadata.get("image_path")
                description = image_doc.page_content

                if image_path and image_path in image_data_dict:
                    try:
                        image_bytes = image_data_dict[image_path]
                        image = Image.open(io.BytesIO(image_bytes))
                        st.sidebar.image(
                            image, caption=description, use_container_width=True
                        )
                    except Exception as e:
                        print(f"Failed to display image {image_path}: {str(e)}")
                        continue

            # Show "Load More" button if there are more images
            remaining = total_images - len(images_to_show)
            if remaining > 0:
                if st.sidebar.button(
                    f"Show {remaining} more images",
                    key=f"load_more_{content['company']}",
                ):
                    st.session_state[state_key] = True
                    st.rerun()
            elif st.session_state[state_key] and total_images > images_per_venue:
                if st.sidebar.button(
                    "Show fewer images", key=f"show_less_{content['company']}"
                ):
                    st.session_state[state_key] = False
                    st.rerun()


def initialize_app():
    st.session_state.initialized = True
    load_dotenv(override=True)
    sys.path.append("..")
    with st.spinner("Initializing retriever..."):
        # Configure FAISS with persistence
        PERSIST_DIRECTORY = os.path.join(os.getcwd(), os.getenv("DATABASE_DIR"))

        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        try:
            venue_metadata = load_venue_metadata()
            retriever = initialize_retriever()
            update_retriever(retriever, venue_metadata)
            st.session_state.retriever = retriever

            # Check existing embeddings
            check_existing_embeddings(st.session_state.retriever.vectorstore)

            st.success("Retriever initialized successfully!")
        except Exception as e:
            raise e


# Main app layout
st.write("Ask questions about weddings and venues!")

# Initialize button
if st.session_state.OPENAI_API_KEY and st.session_state.initialized is False:
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
