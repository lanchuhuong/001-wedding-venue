import asyncio
import io
import logging
import os
import sys
import warnings
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from streamlit_carousel import carousel

from function.llm import get_llm_response
from function.retriever import (
    check_existing_embeddings,
    extract_venues,
    initialize_retriever,
    # load_venue_metadata,
    query_documents,
)

bucket_name = "wedding-venues-001"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)


logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="missing ScriptRunContext!"
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
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if "current_supporting_docs" not in st.session_state:
    st.session_state.current_supporting_docs = None
if "venues_from_responses" not in st.session_state:
    st.session_state.venues_from_responses = set()
if "tmp_img_folder" not in st.session_state:
    st.session_state.tmp_img_folder = TemporaryDirectory()
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
        asyncio.run(display_supporting_info(st.session_state.current_supporting_docs))


# Cache for image data
@st.cache_data(ttl=3600, max_entries=100)  # Limit cache size
def get_cached_image_data(blob_path: str, image_path: str) -> bytes:
    blob = bucket.blob(blob_path)
    # Add image compression if size > 1MB
    blob.download_to_filename(image_path)
    # image_data = blob.download_as_bytes()
    # if len(image_data) > 1_000_000:  # 1MB
    #     image = Image.open(io.BytesIO(image_data))
    #     image.thumbnail((800, 800))  # Resize large images
    #     buffer = io.BytesIO()
    #     image.save(buffer, format="JPEG", quality=85)
    #     return buffer.getvalue()
    # return image_data


async def fetch_single_image_async(
    image_path: str, company_name: str
) -> tuple[str, bytes]:
    image_filename = os.path.basename(image_path)
    full_path = f"processed/adobe_extracted/{company_name}/figures/{image_filename}"
    get_cached_image_data(full_path, image_path)
    # image_data = get_cached_image_data(full_path, image_path)
    # return image_path, image_data


async def preload_images_async(
    company_name: str, image_paths: List[str]
) -> Dict[str, bytes]:
    tasks = [fetch_single_image_async(path, company_name) for path in image_paths]
    results = await asyncio.gather(*tasks)
    # return {path: data for path, data in results if data is not None}


@lru_cache
def get_docs_for_venue(venue: str):
    result_docs = list(
        filter(
            lambda item: (item.metadata["company"] == venue),
            st.session_state.retriever.vectorstore.docstore._dict.values(),
        )
    )
    return result_docs


@lru_cache
def get_image_paths_for_venue(venue: str):
    result_docs = get_docs_for_venue(venue)
    tmp_image_paths = [
        os.path.join(st.session_state.tmp_img_folder.name, doc.metadata["image_path"])
        for doc in result_docs
        if doc.metadata.get("image_path")
    ]
    for path in tmp_image_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return tmp_image_paths


@lru_cache
def get_text_for_venue(venue: str):
    result_docs = get_docs_for_venue(venue)
    return [doc.page_content for doc in result_docs]


@lru_cache
def get_venue_info_for_venue(venue: str):
    result_docs = get_docs_for_venue(venue)
    if len(result_docs) == 0:
        return {}
    result_doc = result_docs[0]
    return {
        "website": result_doc.metadata.get("website"),
        "phone": result_doc.metadata.get("phone"),
        "address": result_doc.metadata.get("address"),
    }


async def display_supporting_info(venues, images_per_venue=3):
    """
    Display venue information with progressive image loading.

    Args:
        results: Dictionary of venue results
        images_per_venue: Number of images to show initially per venue
    """
    if not venues:
        return

    structured_documents = [
        {
            "company": venue,
            "metadata": {
                "image_paths": get_image_paths_for_venue(venue),
                "venue_information": get_venue_info_for_venue(venue),
            },
        }
        for venue in venues
    ]
    with st.sidebar.container():
        for document in structured_documents:
            print(f"document is: {document}")
            company = document["company"]
            metadata = document["metadata"]
            image_paths = metadata["image_paths"]
            venue_info = metadata["venue_information"]
            if image_paths:
                await preload_images_async(company, image_paths)
            with st.expander(company):
                st.subheader("Venue Information")
                if metadata.get("website"):
                    st.markdown(f"🌐 [Website]({venue_info['website']})")

                if metadata.get("phone") and pd.notna(venue_info["phone"]):
                    st.markdown(f"📞 {metadata['phone']}")

                if metadata.get("address"):
                    st.markdown(f"📍 {metadata['address']}")

                if image_paths:
                    st.subheader("Images")
                    carousel_items = [
                        dict(
                            title="",
                            text="",
                            img=img_path,
                        )
                        for i, img_path in enumerate(image_paths)
                    ]
                    carousel(items=carousel_items)

    # for doc_id, content in results.items():
    #     st.sidebar.subheader(f" 💒 {content['company']}")

    #     # Display venue metadata
    #     metadata = next(iter(content["images"] + content["text"])).metadata

    #     with st.sidebar.container():
    #         st.markdown('<div class="venue-info">', unsafe_allow_html=True)

    #         if metadata.get("website"):
    #             st.markdown(f"🌐 [Website]({metadata['website']})")

    #         if metadata.get("phone") and pd.notna(metadata["phone"]):
    #             st.markdown(f"📞 {metadata['phone']}")

    #         if metadata.get("address"):
    #             st.markdown(f"📍 {metadata['address']}")

    #         st.markdown("</div>", unsafe_allow_html=True)

    #     # Handle images with progressive loading
    #     if content["images"]:
    #         st.sidebar.write("🖼️ Venue Images:")

    #         # Get state key for this venue
    #         state_key = f"show_all_images_{content['company']}"
    #         if state_key not in st.session_state:
    #             st.session_state[state_key] = False

    #         # Determine how many images to show
    #         total_images = len(content["images"])
    #         if st.session_state[state_key]:
    #             images_to_show = content["images"]
    #         else:
    #             images_to_show = content["images"][:images_per_venue]

    #         # Collect image paths for the images we'll display
    #         image_paths = [
    #             doc.metadata.get("image_path")
    #             for doc in images_to_show
    #             if doc.metadata.get("image_path")
    #         ]

    #         # Preload selected images
    #         image_data_dict = await preload_images_async(
    #             content["company"], image_paths
    #         )

    #         # Display images
    #         for image_doc in images_to_show:
    #             image_path = image_doc.metadata.get("image_path")
    #             # description = image_doc.page_content

    #             if image_path and image_path in image_data_dict:
    #                 try:
    #                     image_bytes = image_data_dict[image_path]
    #                     image = Image.open(io.BytesIO(image_bytes))
    #                     st.sidebar.image(
    #                         image,
    #                         # caption=description,
    #                         use_container_width=True,
    #                     )
    #                 except Exception as e:
    #                     print(f"Failed to display image {image_path}: {str(e)}")
    #                     continue

    #         # Show "Load More" button if there are more images
    #         remaining = total_images - len(images_to_show)
    #         if remaining > 0:
    #             if st.sidebar.button(
    #                 f"Show {remaining} more images",
    #                 key=f"load_more_{content['company']}_{doc_id}",
    #             ):
    #                 st.session_state[state_key] = True
    #                 st.rerun()
    #         elif st.session_state[state_key] and total_images > images_per_venue:
    #             if st.sidebar.button(
    #                 "Show fewer images", key=f"show_less_{content['company']}_{doc_id}"
    #             ):
    #                 st.session_state[state_key] = False
    #                 st.rerun()


def create_supporting_docs(venues: List[str]) -> dict:
    """
    Create supporting documentation for the specified venues by retrieving their information
    from the vectorstore.

    Args:
        venues (List[str]): List of venue names to retrieve documentation for

    Returns:
        dict: Dictionary with document IDs as keys and venue information as values
    """
    # Get the retriever from session state
    retriever = st.session_state.retriever

    # Initialize the supporting docs dictionary
    supporting_docs = {}

    # Get all documents from vectorstore
    all_docs = retriever.vectorstore.docstore._dict.items()

    # Group documents by doc_id
    for doc_id, doc in all_docs:
        company = doc.metadata.get("company")
        if company in venues:
            # Initialize the venue entry if it doesn't exist
            if doc.metadata["doc_id"] not in supporting_docs:
                supporting_docs[doc.metadata["doc_id"]] = {
                    "company": company,
                    "text": [],
                    "images": [],
                }

            # Add document to appropriate list based on type
            if doc.metadata.get("type") == "image":
                supporting_docs[doc.metadata["doc_id"]]["images"].append(doc)
            else:
                supporting_docs[doc.metadata["doc_id"]]["text"].append(doc)

    return supporting_docs


def initialize_app():
    st.session_state.initialized = True
    load_dotenv(override=True)
    sys.path.append("..")
    with st.spinner("Initializing retriever..."):
        try:
            # venue_metadata = load_venue_metadata()
            retriever = initialize_retriever()
            # update_retriever(retriever, venue_metadata)
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
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])
    if msg["role"] == "assistant" and "supporting_docs" in msg:
        asyncio.run(display_supporting_info(st.session_state.venues_from_responses))


if query := st.chat_input("Ask about wedding venues..."):
    if not st.session_state.retriever:
        st.error("Please initialize the system first!")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        with st.spinner("Searching..."):
            try:
                # Get retrieved documents
                results = query_documents(st.session_state.retriever, query)
                for key, value in results.items():
                    print(f"relevant companies are: {value['company']}")
                # Get LLM response
                llm_response = get_llm_response(
                    query, results, st.session_state.chat_history
                )

                response = st.chat_message("assistant").write_stream(llm_response)

                response = response.encode("utf-8").decode("utf-8")
                venues_from_response = extract_venues(response)
                print(f"venues_from_response are: {venues_from_response}")
                supporting_docs = create_supporting_docs(venues_from_response)
                for key, value in supporting_docs.items():
                    print(f"raw companies are: {value['company']}")
                st.session_state.venues_from_responses.update(venues_from_response)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "supporting_docs": supporting_docs,
                        "venues_from_responses": st.session_state.venues_from_responses,
                    }
                )
                print(
                    f"venues_from_responses are: {st.session_state.venues_from_responses}"
                )
                asyncio.run(
                    display_supporting_info(st.session_state.venues_from_responses)
                )
            except Exception as e:
                st.error(f"Error: {e}")
