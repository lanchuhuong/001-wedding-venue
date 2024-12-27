import asyncio
import os
import re
import uuid
import warnings
from collections import defaultdict
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud import storage
from langchain.schema import Document
from streamlit_carousel import carousel

from function.cloud import list_files
from function.llm import get_llm_response
from function.retriever import (
    extract_venues,
    initialize_retriever,
    query_documents,
)

bucket_name = "wedding-venues-001"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)


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
    st.session_state.venues_from_responses = dict.fromkeys([])
if "tmp_img_folder" not in st.session_state:
    st.session_state.tmp_img_folder = TemporaryDirectory()
if "all_relevant_docs" not in st.session_state:
    st.session_state.all_relevant_docs = []
if "all_relevant_venues" not in st.session_state:
    st.session_state.all_relevant_venues = dict.fromkeys([])

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


async def get_cached_image_data_async(blob_path: str, image_path: str) -> None:
    """Async wrapper for downloading image data"""
    blob = bucket.blob(blob_path)
    # Create a coroutine for the blocking download operation
    await asyncio.to_thread(blob.download_to_filename, image_path)


async def fetch_single_image_async(image_path: str, company_name: str) -> None:
    """Fetch a single image asynchronously"""
    image_filename = os.path.basename(image_path)
    full_path = f"processed/adobe_extracted/{company_name}/figures/{image_filename}"
    await get_cached_image_data_async(full_path, image_path)


async def preload_images_async(company_name: str, image_paths: List[str]) -> None:
    """Concurrently preload all images for a venue"""
    tasks = [fetch_single_image_async(path, company_name) for path in image_paths]
    await asyncio.gather(*tasks)


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
    # result_docs = get_docs_for_venue(venue)
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


@st.cache_resource
def get_text_for_venue(venue: str):
    result_docs = get_docs_for_venue(venue)
    return [doc.page_content for doc in result_docs]


@st.cache_resource
def get_venue_info_for_venue(venue: str):
    result_docs = get_docs_for_venue(venue)
    if len(result_docs) == 0:
        return {}
    result_doc = result_docs[0]
    venue_info = {
        "website": result_doc.metadata.get("venue_information", {}).get("website")
        or result_doc.metadata.get("website"),
        "phone": result_doc.metadata.get("venue_information", {}).get("phone")
        or result_doc.metadata.get("phone"),
        "address": result_doc.metadata.get("venue_information", {}).get("address")
        or result_doc.metadata.get("address"),
    }
    print(f"\n\nvenue_info for {venue}is: {venue_info}\n\n")
    print(f"\n\nresult_doc for {venue}is: {result_doc.metadata}\n\n")
    return venue_info


async def display_supporting_info():
    """
    Display venue information with progressive image loading.

    Args:
        results: Dictionary of venue results
        images_per_venue: Number of images to show initially per venue
    """
    if not st.session_state.all_relevant_venues:
        return

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
    print(len(structured_documents))
    with st.sidebar.container():
        with st.spinner("Loading supporting information..."):
            for document in structured_documents:
                company = document["company"]
                metadata = document["metadata"]
                image_paths = metadata["image_paths"]
                venue_info = metadata["venue_information"]

                if image_paths:
                    await preload_images_async(company, image_paths)
                with st.expander(company):
                    st.subheader("Venue Information")
                    if venue_info.get("website") and pd.notna(venue_info["website"]):
                        st.markdown(f"🌐 [Website]({venue_info['website']})")

                    if venue_info.get("phone") and pd.notna(venue_info["phone"]):
                        st.markdown(f"📞 {venue_info['phone']}")

                    if venue_info.get("address") and pd.notna(venue_info["address"]):
                        st.markdown(f"📍 {venue_info['address']}")

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

                        state_key = f"show_all_images_{company}"
                        if state_key not in st.session_state:
                            st.session_state[state_key] = False

                        venue_image_dict = all_image_paths_by_venue()
                        total_images = len(venue_image_dict[company])

                        remaining = total_images - len(image_paths)
                        if remaining > 0:
                            if st.button(
                                f"Show {remaining} more images",
                                key=f"load_more_{company}_{uuid.uuid4()}",
                            ):
                                for image_path in venue_image_dict[company]:
                                    doc = Document(
                                        page_content="",
                                        metadata={
                                            "image_path": image_path,
                                            "type": "image",
                                            "doc_id": uuid.uuid4(),
                                        },
                                    )
                                    st.session_state.all_relevant_docs.append(doc)
                                asyncio.run(display_supporting_info())
                            # st.session_state[state_key] = False
                            # st.rerun()

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
    # state_key = f"show_all_images_{content['company']}"
    # if state_key not in st.session_state:
    #     st.session_state[state_key] = False

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

    # # Show "Load More" button if there are more images
    # remaining = total_images - len(images_to_show)
    # if remaining > 0:
    #     if st.sidebar.button(
    #         f"Show {remaining} more images",
    #         key=f"load_more_{content['company']}_{doc_id}",
    #     ):
    #         st.session_state[state_key] = True
    #         st.rerun()
    #         elif st.session_state[state_key] and total_images > images_per_venue:
    #             if st.sidebar.button(
    #                 "Show fewer images", key=f"show_less_{content['company']}_{doc_id}"
    #             ):
    #                 st.session_state[state_key] = False
    #                 st.rerun()


def create_supporting_docs(
    venues: List[str], results: dict[str, dict[str, Any]]
) -> dict:
    """
    Create supporting documentation for the specified venues by retrieving their information
    from the vectorstore.

    Args:
        venues (List[str]): List of venue names to retrieve documentation for

    Returns:
        dict: Dictionary with document IDs as keys and venue information as values
    """
    # Get the retriever from session state
    # retriever = st.session_state.retriever

    # Initialize the supporting docs dictionary
    supporting_docs = {}

    # Get all documents from vectorstore
    # all_docs = retriever.vectorstore.docstore._dict.items()
    relevant_docs = [
        doc for doc in results.values() if doc.metadata.get("company") in venues
    ]
    # Group documents by doc_id
    for doc in relevant_docs:
        company = doc.metadata.get("company")
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
    print("loading retriever from google cloud storage...")
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
    # with st.spinner("Loading supporting information..."):
    #     if st.session_state.all_relevant_docs:
    #         asyncio.run(display_supporting_info())

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
        asyncio.run(
            display_supporting_info(list(st.session_state.venues_from_responses.keys()))
        )


def combine_docs(relevant_docs: list[Document], all_relevant_docs: list[Document]):
    all_relevant_doc_ids = [doc.metadata["doc_id"] for doc in all_relevant_docs]
    for doc in relevant_docs:
        if (
            doc.metadata.get("doc_id")
            and doc.metadata["doc_id"] not in all_relevant_doc_ids
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
                st.session_state.all_relevant_venues.update(
                    dict.fromkeys(venues_from_response)
                )

                all_relevant_docs = remove_irrelevant_docs(
                    all_relevant_docs, st.session_state.all_relevant_venues
                )
                st.session_state.all_relevant_docs = all_relevant_docs

                print(f"venues_from_response are: {venues_from_response}")
                print(
                    f"all_relevant_venues are: {list(st.session_state.all_relevant_venues.keys())}"
                )

                # supporting_docs = create_supporting_docs(
                #     venues_from_response, all_relevant_docs
                # )
                # for key, value in supporting_docs.items():
                # print(f"raw companies are: {value['company']}")
                st.session_state.venues_from_responses.update(
                    dict.fromkeys(venues_from_response)
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        # "supporting_docs": supporting_docs,
                        # "venues_from_responses": st.session_state.venues_from_responses,
                    }
                )

                print(
                    f"venues_from_responses are: {list(st.session_state.venues_from_responses.keys())}"
                )
                asyncio.run(
                    display_supporting_info(
                        # st.session_state.venues_from_responses,
                    )
                )
            except Exception as e:
                st.error(f"Error: {e}")
                raise e
