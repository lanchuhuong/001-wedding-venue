import io
import os
import os.path
import re
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Dict, List

import pandas as pd

try:
    import streamlit as st

    STREAMLIT_INSTALLED = True
except ImportError:
    STREAMLIT_INSTALLED = False
from dotenv import find_dotenv, load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from langchain.retrievers import MultiVectorRetriever
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel
from thefuzz import process

from function.cloud import (
    download_directory,
    download_file,
    download_files,
    list_files,
    upload_directory,
    upload_file,
    upload_files,
)
from function.image import process_images

try:
    from function.pdf_loader import adobeLoader, extract_text_from_file_adobe
except ImportError:
    warnings.warn("PDF loader not found. Skipping PDF processing.")
from function.secrets import secrets

load_dotenv(override=True)

path = find_dotenv()
PROJECT_ROOT = os.path.dirname(path)
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, os.getenv("DATABASE_DIR"))
PDF_PATH: Path = Path(PROJECT_ROOT) / Path(os.getenv("PDF_DIR"))


# bucket_name = "wedding-venues-001"
# storage_client = storage.Client()
# bucket = storage_client.bucket(bucket_name)

storage.Client.from_service_account_json
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
    print(f"Error initializing Google Cloud Storage: {str(e)}")
    raise


def download_faiss_from_cloud(cloud_path: str, local_path: str):
    """
    Downloads the FAISS directory from the cloud storage using the `download_files` function.

    Parameters:
        cloud_path (str): Path in the cloud storage where FAISS is stored (e.g., 'faiss_db').
        local_path (str): Local directory to save the downloaded FAISS database.
    """

    download_file(
        "faiss_db/index.faiss", os.path.join(PERSIST_DIRECTORY, "index.faiss")
    )
    download_file("faiss_db/index.pkl", os.path.join(PERSIST_DIRECTORY, "index.pkl"))

    print(f"Downloaded {2} files from '{cloud_path}' to {PERSIST_DIRECTORY}")
    # return results


def upload_faiss_to_cloud(local_path: str, cloud_path: str):
    """
    Uploads the FAISS directory to the cloud storage using the `upload_files` function.

    Parameters:
        local_path (str): Path to the local directory containing FAISS files (e.g., 'faiss_db').
        cloud_path (str): Path in the cloud storage where FAISS will be stored (e.g., 'faiss_db').
    """
    # Recursively gather all files from the local directory
    all_files = [
        (
            os.path.join(root, file),
            os.path.join(
                cloud_path, os.path.relpath(os.path.join(root, file), local_path)
            ),
        )
        for root, _, files in os.walk(local_path)
        for file in files
    ]

    cloud_url = upload_files(all_files)

    print(f"Uploaded {len(cloud_url)} files from '{local_path}' to '{cloud_path}'")
    return cloud_url


def initialize_database(from_cloud=True) -> FAISS:
    """
    Initialize a FAISS database with OpenAI embeddings.
    Loads from disk if exists, creates new if not.

    Returns
    -------
    FAISS
        Initialized FAISS vector store instance with OpenAI embeddings.
    """
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=secrets.OPENAI_API_KEY.get_secret_value(),
    )
    # if not os.path.exists(PERSIST_DIRECTORY):
    print("Fetching database from the cloud...")
    if from_cloud:
        download_faiss_from_cloud(bucket.blob("faiss_db"), PERSIST_DIRECTORY)
    # Try to load
    try:
        vectorstore = FAISS.load_local(
            PERSIST_DIRECTORY, embedding_model, allow_dangerous_deserialization=True
        )
        print(f"Loaded FAISS index from {PERSIST_DIRECTORY}")
        return vectorstore
    except Exception as e:
        print(f"Error loading existing index: {e}")


def cache_resource_if_available(func):
    """Decorator that applies st.cache_resource if Streamlit is installed"""
    if STREAMLIT_INSTALLED:
        return st.cache_resource(func)
    return func


@cache_resource_if_available
def initialize_retriever():
    vectorstore = initialize_database()
    retriever = _initialize_retriever(vectorstore)
    return retriever


def initialize_retriever_from_disk():
    vectorstore = initialize_database()
    retriever = _initialize_retriever(vectorstore)
    return retriever


def _initialize_retriever(vectorstore: FAISS) -> MultiVectorRetriever:
    """
    Initialize a MultiVectorRetriever with the given vectorstore.

    Parameters
    ----------
    vectorstore : FAISS
        The FAISS vectorstore to use for the retriever.

    Returns
    -------
    MultiVectorRetriever
        Initialized retriever instance.
    """
    store = InMemoryStore()
    id_key = "content_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={"k": 49}
    )
    existing_docs = [doc for doc in vectorstore.docstore._dict.values()]
    for doc in existing_docs:
        # Add to docstore using the content_id from metadata
        if id_key in doc.metadata:
            store.mset([(doc.metadata[id_key], doc)])
    return retriever


def add_pdfs_to_retriever(
    venues: Iterable[str | Path], retriever: MultiVectorRetriever, venue_metadata
) -> None:
    """
    Add PDFs to the retriever by processing and embedding their content.

    Parameters
    ----------
    pdfs : Iterable[str | Path]
        Collection of PDF names or paths to process.
    retriever : MultiVectorRetriever
        The retriever to add the processed PDFs to.
    """
    doc_infos = preprocess_documents(venues, venue_metadata)
    add_documents_to_retriever(doc_infos, retriever, venue_metadata)


def remove_pdfs_from_retriever(
    deleted_pdfs: Iterable[str], retriever: MultiVectorRetriever
) -> None:
    """
    Remove PDFs from the retriever's database.

    Parameters
    ----------
    deleted_pdfs : Iterable[str]
        Collection of PDF names to remove.
    retriever : MultiVectorRetriever
        The retriever to remove the PDFs from.
    """
    company_to_idx_mapping = defaultdict(lambda: [])
    for idx, doc in retriever.vectorstore.docstore._dict.items():
        company = doc.metadata["company"]
        company_to_idx_mapping[company].append(idx)

    for pdf in deleted_pdfs:
        retriever.vectorstore.delete(company_to_idx_mapping[pdf])


def get_all_venue_names_on_cloud():
    venue_paths = list_files(r"venues/.*")
    pattern = re.compile("venues/(.*)/.*.pdf")
    venue_names = [pattern.findall(path)[0] for path in venue_paths]
    return venue_names


def update_retriever(retriever: MultiVectorRetriever, venue_metadata) -> None:
    """
    Update the retriever by adding new PDFs and removing deleted ones.

    Parameters
    ----------
    retriever : MultiVectorRetriever
        The retriever to update.
    """
    metadatas = [
        entry.metadata for entry in retriever.vectorstore.docstore._dict.values()
    ]
    all_stored_companies = set(_["company"] for _ in metadatas)

    all_companies = set(get_all_venue_names_on_cloud())

    new_pdfs = all_companies - all_stored_companies
    deleted_pdfs = all_stored_companies - all_companies

    add_pdfs_to_retriever(new_pdfs, retriever, venue_metadata)
    # remove_pdfs_from_retriever(deleted_pdfs, retriever)
    retriever.vectorstore.save_local(PERSIST_DIRECTORY)
    if new_pdfs or deleted_pdfs:
        print(f"all pdfs in {PDF_PATH}: {all_companies}")
        print(f"all pdfs in database: {all_stored_companies}")
        print(f"new pdfs: {new_pdfs}")
        print(f"deleted pdfs: {deleted_pdfs}")


@lru_cache
def get_all_venues():
    all_venues = get_all_venue_names_on_cloud()
    return all_venues


get_all_venues()


class VenueList(BaseModel):
    venues: List[str]


def extract_venues(text: str) -> VenueList:
    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Extract all wedding venue names from the provided text. 
                Return only formal venue names without any additional details or descriptions.""",
            },
            {"role": "user", "content": text},
        ],
        response_format=VenueList,
    )
    unvalidated_venue_list = completion.choices[0].message.parsed
    validated_venue_list = []
    for venue in unvalidated_venue_list.venues:
        best_match, score = process.extractOne(venue, get_all_venues(), score_cutoff=90)
        validated_venue_list.append(best_match)
    return validated_venue_list


def query_documents(retriever: MultiVectorRetriever, query: str) -> list[Document]:
    """
    Query documents from the retriever using the given query string with source diversity.

    Parameters
    ----------
    retriever : MultiVectorRetriever
        The retriever to query documents from.
    query : str
        The search query string.

    Returns
    -------
    Dict[str, Dict[str, List[Document]]]
        Dictionary containing query results organized by document ID.
    """
    MAX_RETURNED_VENUES = 7
    vectorstore = retriever.vectorstore
    # Get more initial documents to allow for filtering
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        similar_docs_with_score = vectorstore.similarity_search_with_relevance_scores(
            query, k=49
        )

    results = []
    for doc, score in similar_docs_with_score:
        if score > 0:
            results.append(doc)
    all_venues_with_duplicates = [doc.metadata["company"] for doc in results]

    # Remove duplicates while preserving order
    unique_venues = list(dict.fromkeys(all_venues_with_duplicates))

    top_relevant_venues = unique_venues[:MAX_RETURNED_VENUES]

    top_results = [
        doc for doc in results if doc.metadata["company"] in top_relevant_venues
    ]
    return top_results

    # Group documents by source (doc_id)
    docs_by_source = {}
    for doc, score in similar_docs_with_score:
        print(score, doc.metadata["company"], doc.metadata["type"])
        doc_id = doc.metadata["doc_id"]
        if doc_id not in docs_by_source:
            docs_by_source[doc_id] = []
        docs_by_source[doc_id].append((doc, score))

    # Get diverse documents
    results: dict[str, dict[str, Any]] = {}
    score_threshold = 0.0
    processed_docs = 0

    # Sort sources by their best score
    sorted_sources = list(
        sorted(
            docs_by_source.items(),
            key=lambda x: min(
                score for _, score in x[1]
            ),  # Get best score for each source
        )
    )
    print("sorted sources:")
    print(
        (score, doc.metadata["company"], doc.metadata["type"])
        for doc, score in sorted_sources
    )
    # Process each source
    for doc_id, docs_and_scores in sorted_sources:
        # Skip if we already have enough documents
        if processed_docs >= 7:
            break
        # Sort documents within this source by score
        docs_and_scores.sort(key=lambda x: x[1])  # Sort by score

        # Take up to 3 documents from this source
        relevant_docs = docs_and_scores[:3]

        # Filter by score threshold and process
        for doc, score in relevant_docs:
            if score <= score_threshold:
                continue

            if doc_id not in results:
                results[doc_id] = {
                    "company": doc.metadata["company"],
                    "text": [],
                    "images": [],
                }
            print(f"relevant PDFs: {results[doc_id]['company']}")

            if doc.metadata["type"] == "text":
                results[doc_id]["text"].append(doc)
            else:
                results[doc_id]["images"].append(doc)

            processed_docs += 1

    return results


def load_venue_metadata():
    """
    Load venue metadata from Excel file into a dictionary.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file containing venue metadata

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with venue names as keys and their metadata as values
    """

    excel_blob = bucket.blob("Wedding Venues.xlsx")
    excel_content = excel_blob.download_as_bytes()
    df = pd.read_excel(io.BytesIO(excel_content))
    metadata_dict = {}
    for _, row in df.iterrows():
        metadata_dict[row["Venue name"]] = {
            "phone": row.get("Phone", ""),
            "address": row.get("Location", ""),
            "website": row.get("Website ", ""),
        }
    return metadata_dict


def get_venue_images_from_cloud(venue, destination_folder):
    """
    Downloads images for a venue from the cloud to the specified destination folder.

    Parameters
    ----------
    venue : str
        The venue name.
    destination_folder : Path or str
        The destination folder where the images will be saved.
    """
    destination_folder = Path(destination_folder)
    images = list_files(f"processed/adobe_extracted/{venue}/figures/.*")

    # Ensure the destination folder exists
    destination_folder.mkdir(parents=True, exist_ok=True)

    download_files(
        images,
        [str(destination_folder / Path(image).name) for image in images],
    )


def file_exists(name):
    client = storage.Client()
    bucket = client.bucket("wedding-venues-001")

    blob1 = bucket.blob(name)
    blob2 = bucket.blob("/" + name)
    return blob1.exists() or blob2.exists()


def preprocess_document(
    venue: str, venue_metadata: Dict[str, Dict[str, Any]]
) -> dict[str, Any]:
    """
    preprocess_document that includes venue metadata.
    """
    with (
        NamedTemporaryFile(suffix=".pdf") as temp_pdf_file,
        NamedTemporaryFile(suffix=".zip") as temp_zip_file,
        TemporaryDirectory() as temp_output_dir,
    ):
        is_processed = file_exists(f"processed/adobe_result/{venue}/sdk.zip")
        extracted_exists = file_exists(
            f"processed/adobe_extracted/{venue}/structuredData.json"
        )
        # is_processed = (
        #     len(list_files(filter=rf"processed/adobe_result/{venue}/sdk.zip")) > 0
        # )
        # extracted_exists = (
        #     len(
        #         list_files(
        #             filter=rf"processed/adobe_extracted/{venue}/structuredData.json"
        #         )
        #     )
        #     > 0
        # )

        text_embedding_exists = False
        retriever = initialize_retriever_from_disk()
        docs = retriever.vectorstore.docstore._dict.values()
        for doc in docs:
            if (doc.metadata["company"] == venue) and (doc.metadata["type"] == "text"):
                text_embedding_exists = True
                break

        if not is_processed:
            # No ZIP file, process the PDF with Adobe
            print(f"searching for {venue}.pdf on Google Cloud...")
            cloud_venue_path = list_files(filter=rf"venues/{venue}/.*.pdf")
            if not cloud_venue_path:
                print(f"no PDF found for {venue}")
                return None
            cloud_venue_path = cloud_venue_path[0]
            print(f"downloading {venue}.pdf from Google Cloud...")
            download_file(cloud_venue_path, temp_pdf_file.name)
            print(f"sending {venue}.pdf to Adobe...")
            try:
                adobeLoader(temp_pdf_file.name, temp_zip_file.name)
            except Exception as e:
                print(f"error with Adobe operation: {e}")
                return None
            print(f"extracting text from Adobe ZIP for {venue}...")
            text_content = extract_text_from_file_adobe(
                temp_zip_file.name, temp_output_dir
            )

        elif extracted_exists and not text_embedding_exists:
            # ZIP file exists, and the extracted folder exists
            print(f"Extracted folder already exists for {venue}. Downloading...")
            extracted_folder_path = os.path.join(
                os.getcwd(), f"processed/adobe_extracted/{venue}/"
            )
            download_directory(
                venue, temp_output_dir, verbose=True
            )  # Download the extracted folder
            print(f"Running extract_text_from_file_adobe for {venue}...")
            text_content = extract_text_from_file_adobe(venue, temp_output_dir)

        elif not text_embedding_exists:
            # ZIP exists but no extracted folder
            print(f"found sdk.zip for {venue}. downloading...")
            zip_file_path = f"processed/adobe_result/{venue}/sdk.zip"
            extracted_folder_path = os.path.join(
                os.getcwd(), f"processed/adobe_extracted/{venue}/"
            )
            download_file(zip_file_path, temp_zip_file.name)
            print(f"extracting text from ZIP for {venue}...")
            text_content = extract_text_from_file_adobe(
                temp_zip_file.name, temp_output_dir
            )
        else:
            return None
        image_descriptions = process_images(
            venue, os.path.join(temp_output_dir, "figures"), retriever
        )
        # extracted_figure_folder = Path(temp_output_dir) / "figures"
        # get_venue_images_from_cloud(venue, extracted_figure_folder)
        # if not extracted_figure_folder.exists():
        #     print(f"no images found for {venue}.pdf")
        #     image_descriptions = []
        # else:
        #     print(f"generating image descriptions for {venue}.pdf")
        #     image_descriptions = generate_image_descriptions(
        #         base_dir=extracted_figure_folder,
        #         venue=venue,
        #     )

        print(f"uploading extracted folder for {venue} to Google Cloud...")
        upload_directory(temp_output_dir, f"/processed/adobe_extracted/{venue}/")

    doc_id = str(uuid.uuid4())
    # Include venue metadata in document_info
    venue_info = venue_metadata.get(venue, {})
    document_info = {
        "doc_id": doc_id,
        "text_content": text_content,
        "image_descriptions": image_descriptions,
        "metadata": venue_info,
    }

    return document_info


def add_documents_to_retriever(
    documents: dict[str, dict[str, Any]],
    retriever: MultiVectorRetriever,
    venue_metadata: Dict[str, Dict[str, Any]],
) -> None:
    """
    add_documents_to_retriever that includes venue metadata.
    """
    id_key = "content_id"

    for pdf_name, doc_info in documents.items():
        # Get venue metadata
        venue_info = venue_metadata.get(pdf_name, {})
        text_docs = [
            Document(
                page_content=doc_info["text_content"],
                metadata={
                    id_key: f"{doc_info['doc_id']}_text",
                    "doc_id": doc_info["doc_id"],
                    "company": pdf_name,
                    "type": "text",
                    "website": venue_info.get("website", ""),
                    "address": venue_info.get("address", ""),
                    "phone": venue_info.get("phone", ""),
                },
            )
        ]

        # Create image documents with metadata
        image_ids = [
            f"{doc_info['doc_id']}_image_{i}"
            for i in range(len(doc_info["image_descriptions"]))
        ]
        image_docs = [
            Document(
                page_content=item["description"],
                metadata={
                    id_key: image_ids[i],
                    "doc_id": doc_info["doc_id"],
                    "company": pdf_name,
                    "type": "image",
                    "image_path": item["image_path"],
                    "website": venue_info.get("website", ""),
                    "address": venue_info.get("address", ""),
                    "phone": venue_info.get("phone", ""),
                },
            )
            for i, item in enumerate(doc_info["image_descriptions"])
        ]

        all_docs = text_docs + image_docs
        retriever.vectorstore.add_documents(all_docs)

        original_data = [(doc.metadata[id_key], doc) for doc in all_docs]
        retriever.docstore.mset(original_data)

        print(f"Processed document: {pdf_name}")


def preprocess_documents(
    venues: Iterable[str], venue_metadata
) -> dict[str, dict[str, Any]]:
    """
    Preprocess PDFs by extracting text and generating image descriptions.

    Parameters
    ----------
    pdf_paths : Iterable[str | Path]
        Collection of PDF file paths to process.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing preprocessed document information including text content
        and image descriptions.
    """

    new_documents: dict[str, dict[str, Any]] = {}
    retriever = initialize_retriever()
    # for venue in tqdm(venues):
    try:
        for venue in venues:
            document_info = preprocess_document(venue, venue_metadata)
            if document_info is None:
                continue

            add_documents_to_retriever(
                {venue: document_info}, retriever, venue_metadata
            )
            retriever.vectorstore.save_local(PERSIST_DIRECTORY)
            new_documents[venue] = document_info
    finally:
        print("uploading retriever to cloud")
        upload_retriever_to_cloud()

    return new_documents


#         print(f"Processed document: {pdf_name}")
def check_existing_embeddings(vectorstore: FAISS) -> None:
    """
    Print information about existing embeddings in the vectorstore.

    Parameters
    ----------
    vectorstore : FAISS
        The vectorstore to check embeddings from.
    """
    existing_docs = vectorstore.docstore._dict.values()

    print(f"Total documents in vectorstore: {len(existing_docs)}")
    print("Existing document companies:")
    for doc in existing_docs:
        print(f"- ({doc.metadata.get('type')}){doc.metadata.get('company', 'Unknown')}")


def upload_retriever_to_cloud() -> None:
    upload_file(os.path.join(PERSIST_DIRECTORY, "index.faiss"), "faiss_db/index.faiss")
    upload_file(os.path.join(PERSIST_DIRECTORY, "index.pkl"), "faiss_db/index.pkl")
