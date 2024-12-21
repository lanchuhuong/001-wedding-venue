import os
import os.path
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Any

import matplotlib.pyplot as plt
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from PIL import Image
from tqdm import tqdm

from function.cloud import download_file, list_files, upload_directory
from function.pdf_loader import adobeLoader, extract_text_from_file_adobe
from function.process_image import generate_image_descriptions

sys.path.append("..")

PERSIST_DIRECTORY: str = os.getenv("DATABASE_DIR")
PDF_PATH: Path = Path(os.getenv("PDF_DIR"))

# print(os.path.exists(PERSIST_DIRECTORY))


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
        api_key=os.getenv("OPENAI_API_KEY"),
        # st.session_state.OPENAI_API_KEY
    )

    # Try to load existing index
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            vectorstore = FAISS.load_local(
                PERSIST_DIRECTORY, embedding_model, allow_dangerous_deserialization=True
            )
            print(f"Loaded existing FAISS index from {PERSIST_DIRECTORY}")
            return vectorstore
        except Exception as e:
            print(f"Error loading existing index: {e}")

    # Create new if doesn't exist
    dummy_doc = Document(
        page_content="dummy document", metadata={"company": "dummy", "type": "text"}
    )
    vectorstore = FAISS.from_documents(
        documents=[dummy_doc],
        embedding=embedding_model,
    )

    # Save to disk
    vectorstore.save_local(PERSIST_DIRECTORY)
    print(f"Created new FAISS index in {PERSIST_DIRECTORY}")

    return vectorstore


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
        vectorstore=vectorstore, docstore=store, id_key=id_key, search_kwargs={"k": 4}
    )
    existing_docs = [doc for doc in vectorstore.docstore._dict.values()]
    for doc in existing_docs:
        # Add to docstore using the content_id from metadata
        if id_key in doc.metadata:
            store.mset([(doc.metadata[id_key], doc)])
    return retriever


def add_pdfs_to_retriever(
    pdfs: Iterable[str | Path], retriever: MultiVectorRetriever
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
    pdf_paths = [PDF_PATH / f"{pdf}.pdf" for pdf in pdfs]
    doc_infos = preprocess_documents(pdf_paths)
    add_documents_to_retriever(doc_infos, retriever)


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


def update_retriever(retriever: MultiVectorRetriever) -> None:
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

    all_companies = set(
        path.name.replace(".pdf", "") for path in PDF_PATH.glob("*.pdf")
    )
    all_companies = {path.name for path in Path("venues").glob("*")}

    new_pdfs = all_companies - all_stored_companies
    deleted_pdfs = all_stored_companies - all_companies

    add_pdfs_to_retriever(new_pdfs, retriever)
    remove_pdfs_from_retriever(deleted_pdfs, retriever)
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
    similar_docs = retriever.invoke(query)
    print(f"similar document: {similar_docs}")
    results: dict[str, dict[str, Any]] = {}

    for doc in similar_docs:
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

    for doc_id, content in results.items():
        print(f"Company: {content['company']}")

        for text_doc in content["text"]:
            print(f"Text: {text_doc.page_content}\n")

        for image_doc in content["images"]:
            image_path = image_doc.metadata.get("image_path")
            description = image_doc.page_content
            print(f"Image Description: {description}")
            if image_path:
                try:
                    im = Image.open(image_path)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(im)
                    plt.axis("off")
                    plt.show()
                except Exception as e:
                    print(f"Error displaying image: {e}")

    return results


def add_documents_to_retriever(
    documents: dict[str, dict[str, Any]], retriever: MultiVectorRetriever
) -> None:
    """
    Add processed documents to the retriever.

    Parameters
    ----------
    documents : Dict[str, Dict[str, Any]]
        Dictionary containing processed document information.
    retriever : MultiVectorRetriever
        The retriever to add documents to.
    """
    id_key = "content_id"

    for pdf_name, doc_info in documents.items():
        text_ids = [
            f"{doc_info['doc_id']}_text_{i}"
            for i in range(len(doc_info["text_content"]))
        ]
        text_docs = [
            Document(
                page_content=row["text"],
                metadata={
                    id_key: text_ids[i],
                    "doc_id": doc_info["doc_id"],
                    "company": pdf_name,
                    "type": "text",
                },
            )
            for i, row in enumerate(doc_info["text_content"])
        ]

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
                },
            )
            for i, item in enumerate(doc_info["image_descriptions"])
        ]

        all_docs = text_docs + image_docs
        retriever.vectorstore.add_documents(all_docs)

        original_data = [(doc.metadata[id_key], doc) for doc in all_docs]
        retriever.docstore.mset(original_data)

        print(f"Processed document: {pdf_name}")


def preprocess_documents(venues: Iterable[str]) -> dict[str, dict[str, Any]]:
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
        is_processed = len(list_files(filter=rf"/{venue}/structuredData.json")) > 0
        if not is_processed:
            document_info = preprocess_document(venue)
            new_documents[venue] = document_info

    return new_documents


def preprocess_document(venue: str) -> dict[str, Any]:
    with (
        NamedTemporaryFile(suffix=".pdf") as temp_pdf_file,
        NamedTemporaryFile(suffix=".zip") as temp_zip_file,
        TemporaryDirectory() as temp_output_dir,
    ):
        cloud_venue_path = list_files(filter=rf"venues/{venue}.pdf")[0]
        download_file(cloud_venue_path, temp_pdf_file.name)
        adobeLoader(temp_pdf_file.name, temp_zip_file.name)
        text_content = extract_text_from_file_adobe(
            temp_zip_file.name, temp_output_dir.name
        )

        extracted_figure_folder = Path(temp_output_dir.name) / "figures"
        if not extracted_figure_folder.exists():
            image_descriptions = []
        else:
            image_descriptions = generate_image_descriptions(
                base_dir=extracted_figure_folder,
                venue=venue,
            )
            doc_id = str(uuid.uuid4())
        upload_directory(temp_output_dir.name, f"/processed/adobe_extracted/{venue}/")
    document_info = {
        "doc_id": doc_id,
        "text_content": text_content.to_dict("records"),
        "image_descriptions": image_descriptions,
    }

    return document_info
    # output_base_zip_path = Path("data/processed/adobe_result/")
    # output_base_extract_folder = Path("data/processed/adobe_extracted/")
    # print(f"processing {pdf_path}")
    # pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    # output_zip_path = os.path.join(output_base_zip_path, pdf_name, "sdk.zip")
    # output_zipextract_folder = os.path.join(output_base_extract_folder, pdf_name)

    # if not os.path.exists(
    #     os.path.join(output_zipextract_folder, "structuredData.json")
    # ):
    #     print(f"loading {pdf_name} to adobe pdf services...")
    #     adobeLoader(
    #         pdf_path,
    #         output_zip_path=output_zip_path,
    #         client_id=client_id,
    #         client_secret=client_secret,
    #     )
    # df = extract_text_from_file_adobe(output_zip_path, output_zipextract_folder)
    # df["company"] = pdf_name
    # text_content = (
    #     df.groupby("company")["text"].apply(lambda x: "\n".join(x)).reset_index()
    # )
    # extracted_figure_folder = Path(output_zipextract_folder) / "figures"
    # if not extracted_figure_folder.exists():
    #     image_descriptions = []
    # else:
    #     image_descriptions = generate_image_descriptions(
    #         base_dir=extracted_figure_folder,
    #         pdf_name=pdf_name,
    #         output_file=os.path.join(
    #             output_base_extract_folder, f"{pdf_name}_descriptions.json"
    #         ),
    #     )

    # doc_id = str(uuid.uuid4())

    # document_info = {
    #     "doc_id": doc_id,
    #     "text_content": text_content.to_dict("records"),
    #     "image_descriptions": image_descriptions,
    # }

    # return document_info


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
