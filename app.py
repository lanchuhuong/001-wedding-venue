import concurrent.futures
import logging
import os
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from langchain.schema import Document
from streamlit_carousel import carousel

from function.cloud import list_files
from function.llm import get_llm_response
from function.retriever import (
    extract_venues,
    initialize_retriever,
    query_documents,
)

# Set page config
st.set_page_config(page_title="Chat Document", page_icon="🔍", layout="wide")

try:
    # Initialize storage_client as None first
    storage_client = None

    # For local development
    if os.path.exists("turing-guard-444623-s7-2cd0a98f8177.json"):
        storage_client = storage.Client()

    # For Streamlit Cloud
    elif "gcp_service_account" in st.secrets:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        storage_client = storage.Client(
            credentials=credentials,
            project=st.secrets["gcp_service_account"]["project_id"],
        )

    # Check if we successfully got a client
    if storage_client is None:
        raise Exception("No valid credentials found")

    # Initialize bucket
    bucket_name = "wedding-venues-001"
    bucket = storage_client.bucket(bucket_name)

except Exception as e:
    st.error(f"Error initializing Google Cloud Storage: {str(e)}")
    # You might want to raise the exception here depending on your error handling needs
    raise e

logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="missing ScriptRunContext!"
)


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
if "all_relevant_docs" not in st.session_state:
    st.session_state.all_relevant_docs = []
if "all_relevant_venues" not in st.session_state:
    st.session_state.all_relevant_venues = set()

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


def download_single_image(blob_path: str, image_path: str) -> None:
    """Download a single image if it doesn't exist locally"""
    if os.path.exists(image_path):
        return
    blob = bucket.blob(blob_path)
    blob.download_to_filename(image_path)


def preload_images(company_name: str, image_paths: List[str]) -> None:
    """Download multiple images in parallel using ThreadPoolExecutor"""

    def download_image(image_path: str) -> None:
        image_filename = os.path.basename(image_path)
        full_path = f"processed/adobe_extracted/{company_name}/figures/{image_filename}"
        download_single_image(full_path, image_path)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_image, path) for path in image_paths]
        concurrent.futures.wait(futures)


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
def all_image_paths_by_venue() -> dict[str, list[str]]:
    """
    Organizes image filepaths into a dictionary keyed by venue name using regex.
    """
    filepaths = list_files(r"processed/adobe_extracted/.*/figures/.*")
    venue_images = defaultdict(list)

    pattern = r"adobe_extracted/([^/]+)/figures/"

    for path in filepaths:
        match = re.search(pattern, path)
        if match:
            venue_name = match.group(1)
            venue_images[venue_name].append(path)

    return dict(venue_images)


@st.cache_resource
def get_image_paths_for_venue(venue: str, _relevant_docs: list[Document]):
    venue_relevant_docs = filter(
        lambda doc: doc.metadata["company"] == venue, _relevant_docs
    )
    tmp_image_paths = [
        os.path.join(st.session_state.tmp_img_folder.name, doc.metadata["image_path"])
        for doc in venue_relevant_docs
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


def display_supporting_info():
    """
    Display venue information with parallel image loading.
    """
    if not st.session_state.all_relevant_venues:
        return
    t = time.time()
    structured_documents = [
        {
            "company": venue,
            "metadata": {
                "image_paths": get_image_paths_for_venue(
                    venue, st.session_state.all_relevant_docs
                ),
                "venue_information": get_venue_info_for_venue(venue),
            },
        }
        for venue in st.session_state.all_relevant_venues
    ]
    print(f"time to get image paths: {time.time() - t}")

    with st.sidebar.container():
        with st.spinner("Downloading supporting information..."):
            for document in structured_documents:
                company = document["company"]
                metadata = document["metadata"]
                image_paths = metadata["image_paths"]
                venue_info = metadata["venue_information"]
                website = metadata.get("website") or venue_info.get("website")
                if not (website and pd.notna(website) and website != "nan"):
                    website = None
                phone = metadata.get("phone") or venue_info.get("phone")
                if not (phone and pd.notna(phone) and phone != "nan"):
                    phone = None
                address = metadata.get("address") or venue_info.get("address")
                if not (address and pd.notna(address) and address != "nan"):
                    address = None
                if image_paths:
                    preload_images(company, image_paths)
                with st.expander(company):
                    st.subheader("Venue Information")
                    if website:
                        st.markdown(f"🌐 [Website]({website})")

                    if phone:
                        st.markdown(f"📞 {phone}")

                    if address:
                        st.markdown(f"📍 {address}")

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


def create_supporting_docs(
    venues: List[str], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Create supporting documentation for the specified venues by retrieving their information
    from the vectorstore.

    Args:
        venues (List[str]): List of venue names to retrieve documentation for

    Returns:
        dict: Dictionary with document IDs as keys and venue information as values
    """
    # Get the retriever from session state

    # Initialize the supporting docs dictionary
    supporting_docs = {}

    # Get all documents from vectorstore
    relevant_docs = [
        doc for doc in results.values() if doc.metadata.get("company") in venues
    ]
    # Group documents by doc_id
    for doc in relevant_docs:
        company = doc.metadata.get("company")
        # Initialize the venue entry if it doesn't exist
        if doc.metadata.get("doc_id") and (
            doc.metadata["doc_id"] not in supporting_docs
        ):
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
    with st.spinner("Initializing retriever..."):
        try:
            all_image_paths_by_venue()

            retriever = initialize_retriever()
            st.session_state.retriever = retriever

            st.success("Retriever initialized successfully!")
        except Exception as e:
            raise e


# Sidebar for API key
with st.sidebar:
    st.title("Supporting Information")

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
        display_supporting_info()


def combine_docs(relevant_docs: list[Document], all_relevant_docs: list[Document]):
    all_relevant_doc_ids = [doc.metadata["doc_id"] for doc in all_relevant_docs]
    for doc in relevant_docs:
        if doc.metadata.get("doc_id") and (
            doc.metadata["doc_id"] not in all_relevant_doc_ids
        ):
            all_relevant_docs.append(doc)
    return all_relevant_docs


def remove_irrelevant_docs(
    all_relevant_docs: list[Document], all_relevant_venues: set[str]
):
    return [
        doc
        for doc in all_relevant_docs
        if doc.metadata["company"] in all_relevant_venues
    ]


if query := st.chat_input("Ask about wedding venues..."):
    if not st.session_state.retriever:
        st.error("Please initialize the system first!")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        with st.spinner("Searching..."):
            try:
                relevant_docs = query_documents(st.session_state.retriever, query)
                all_relevant_docs = combine_docs(
                    relevant_docs, st.session_state.all_relevant_docs
                )
                print("all_relevant_docs:")
                for doc in all_relevant_docs:
                    print(
                        " - ",
                        doc.metadata["company"],
                        doc.metadata["type"],
                        doc.metadata.get("image_path"),
                    )

                llm_response = get_llm_response(
                    query, all_relevant_docs, st.session_state.chat_history
                )

                response = st.chat_message("assistant").write_stream(llm_response)

                response = response.encode("utf-8").decode("utf-8")
                venues_from_response = extract_venues(response)
                st.session_state.all_relevant_venues.update(venues_from_response)

                all_relevant_docs = remove_irrelevant_docs(
                    all_relevant_docs, st.session_state.all_relevant_venues
                )
                st.session_state.all_relevant_docs = all_relevant_docs

                print(f"venues_from_response are: {venues_from_response}")
                print(
                    f"all_relevant_venues are: {st.session_state.all_relevant_venues}"
                )

                st.session_state.venues_from_responses.update(venues_from_response)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )
                print(
                    f"venues_from_responses are: {st.session_state.venues_from_responses}"
                )
                display_supporting_info()
            except Exception as e:
                st.error(f"Error: {e}")
                raise e
