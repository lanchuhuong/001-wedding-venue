import io
import os
import os.path
import re
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from google.cloud import storage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from PIL import Image
from tqdm import tqdm

from function.cloud import (
    download_file,
    download_files,
    list_files,
    upload_directory,
    upload_file,
    upload_files,
)
from function.pdf_loader import adobeLoader, extract_text_from_file_adobe
from function.process_image import generate_image_descriptions
from function.secrets import secrets

load_dotenv(override=True)

path = find_dotenv()
PROJECT_ROOT = os.path.dirname(path)
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, os.getenv("DATABASE_DIR"))
PDF_PATH: Path = Path(PROJECT_ROOT) / Path(os.getenv("PDF_DIR"))

bucket_name = "wedding-venues-001"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)


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


def initialize_database() -> FAISS:
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


def initialize_retriever() -> MultiVectorRetriever:
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
        vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={"k": 7}
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
    return venue_names[:2]


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


def query_documents(
    retriever: MultiVectorRetriever, query: str
) -> dict[str, dict[str, list[Document]]]:
    """
    Query documents from the retriever using the given query string.

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
    vectorstore = retriever.vectorstore
    similar_docs_with_score = vectorstore.similarity_search_with_score(query, k=7)

    # similar_docs = retriever.invoke(query)
    # print(f"similar document: {similar_docs}")
    results: dict[str, dict[str, Any]] = {}
    score_threshold = 0.7
    for doc, score in similar_docs_with_score:
        print(f"doc_id: {doc.metadata['doc_id']}, score: {score}")

        if score < score_threshold:
            continue
        doc_id = doc.metadata["doc_id"]
        if doc_id not in results:
            results[doc_id] = {
                "company": doc.metadata["company"],
                "text": [],
                "images": [],
            }

        if doc.metadata["type"] == "text":
            results[doc_id]["text"].append(doc)
        else:
            results[doc_id]["images"].append(doc)

    # for doc_id, content in results.items():
    #     for image_doc in content["images"]:
    #         image_path = image_doc.metadata.get("image_path")
    #         # description = image_doc.page_content
    # print(f"Image Description: {description}")
    # if image_path:
    #     try:
    #         im = Image.open(image_path)
    #         plt.figure(figsize=(10, 8))
    #         plt.imshow(im)
    #         plt.axis("off")
    #         plt.show()
    #     except Exception:
    #         print("Error displaying image")

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
    # Assuming 'venue_name' is the column that matches the folder names
    metadata_dict = {}
    for _, row in df.iterrows():
        metadata_dict[row["Venue name"]] = {
            "phone": row.get("Phone", ""),
            "address": row.get("Location", ""),
            "website": row.get("Website ", ""),
        }
    return metadata_dict


def get_venue_images_from_cloud(venue, destination_folder):
    images = list_files(f"/processed/adobe_extracted/{venue}/figures/.*")
    download_files(images, [destination_folder + "/" + image for image in images])


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
        print(f"searching for {venue}.pdf on google cloud...")
        cloud_venue_path = list_files(filter=rf"venues/{venue}/.*.pdf")
        if not cloud_venue_path:
            print(f"no pdf found for {venue}")
            return None
        cloud_venue_path = cloud_venue_path[0]
        print(f"downloading {venue}.pdf from google cloud...")
        download_file(cloud_venue_path, temp_pdf_file.name)
        print(f"sending {venue}.pdf to Adobe...")
        adobeLoader(temp_pdf_file.name, temp_zip_file.name)
        print("extracting text from pdf...")
        text_content = extract_text_from_file_adobe(temp_zip_file.name, temp_output_dir)

        extracted_figure_folder = Path(temp_output_dir) / "figures"
        get_venue_images_from_cloud(venue, extracted_figure_folder)
        if not extracted_figure_folder.exists():
            print(f"no images found for {venue}.pdf")
            image_descriptions = []
        else:
            print(f"generating image descriptions for {venue}.pdf")
            image_descriptions = generate_image_descriptions(
                base_dir=extracted_figure_folder,
                venue=venue,
            )
        print("uploading adobe_extracted_directory to google cloud")
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
                    # Add any other metadata fields
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

    for venue in tqdm(venues):
        is_processed = (
            len(list_files(filter=rf"venues/{venue}/structuredData.json")) > 0
        )
        if not is_processed:
            document_info = preprocess_document(venue, venue_metadata)
            if document_info is None:
                continue
            new_documents[venue] = document_info

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
    upload_file(
        os.join(PERSIST_DIRECTORY, "faiss_db/index.faiss"), "faiss_db/index.faiss"
    )
    upload_file(os.join(PERSIST_DIRECTORY, "faiss_db/index.pkl"), "faiss_db/index.pkl")
