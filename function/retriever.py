import os
import os.path
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chromadb
import matplotlib.pyplot as plt
from chromadb.config import Settings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from PIL import Image

from function.pdf_loader import adobeLoader, extract_text_from_file_adobe
from function.process_image import generate_image_descriptions

sys.path.append("..")

PERSIST_DIRECTORY: str = "chroma_db"
PDF_PATH: Path = Path("data/test_pdf/")


def initialize_database() -> Chroma:
    """
    Initialize a persistent Chroma database with OpenAI embeddings.

    Returns
    -------
    Chroma
        Initialized Chroma vector store instance with OpenAI embeddings.
    """
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY")
    )

    chroma_client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY,
        # settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name="multimodal_docs",
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )

    return vectorstore


def _initialize_retriever(vectorstore: Chroma) -> MultiVectorRetriever:
    """
    Initialize a MultiVectorRetriever with the given vectorstore.

    Parameters
    ----------
    vectorstore : Chroma
        The Chroma vectorstore to use for the retriever.

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
    # Reconstruct the docstore from vectorstore
    existing_docs = vectorstore.get()
    for i, doc_id in enumerate(existing_docs["ids"]):
        metadata = existing_docs["metadatas"][i]
        content = existing_docs["documents"][i]

        # Reconstruct the Document object
        doc = Document(page_content=content, metadata=metadata)

        # Add to docstore using the content_id from metadata
        if id_key in metadata:
            store.mset([(metadata[id_key], doc)])
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
    all_stored_companies = set(
        _["company"] for _ in retriever.vectorstore.get()["metadatas"]
    )
    company_ids = {
        venue: retriever.vectorstore.get(where={"company": venue})["ids"]
        for venue in all_stored_companies
    }
    for pdf in deleted_pdfs:
        retriever.vectorstore.delete(company_ids[pdf])


def update_retriever(retriever: MultiVectorRetriever) -> None:
    """
    Update the retriever by adding new PDFs and removing deleted ones.

    Parameters
    ----------
    retriever : MultiVectorRetriever
        The retriever to update.
    """
    all_stored_companies = set(
        _["company"] for _ in retriever.vectorstore.get()["metadatas"]
    )

    all_companies = set(
        path.name.replace(".pdf", "") for path in PDF_PATH.glob("*.pdf")
    )

    new_pdfs = all_companies - all_stored_companies
    deleted_pdfs = all_stored_companies - all_companies

    add_pdfs_to_retriever(new_pdfs, retriever)
    remove_pdfs_from_retriever(deleted_pdfs, retriever)

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


def preprocess_documents(pdf_paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
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
    output_base_zip_path = "data/processed/adobe_result/"
    output_base_extract_folder = "data/processed/adobe_extracted/"
    output_goodimages_folder = "data/processed/good_figures/"

    new_documents: dict[str, dict[str, Any]] = {}

    for pdf_path in pdf_paths:
        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
        output_zip_path = os.path.join(output_base_zip_path, pdf_name, "sdk.zip")
        output_zipextract_folder = os.path.join(output_base_extract_folder, pdf_name)
        client_id = os.getenv("ADOBE_CLIENT_ID")
        client_secret = os.getenv("ADOBE_CLIENT_SECRET")

        adobeLoader(
            pdf_path,
            output_zip_path=output_zip_path,
            client_id=client_id,
            client_secret=client_secret,
        )
        df = extract_text_from_file_adobe(output_zip_path, output_zipextract_folder)
        df["company"] = pdf_name
        text_content = (
            df.groupby("company")["text"].apply(lambda x: "\n".join(x)).reset_index()
        )

        image_descriptions = generate_image_descriptions(
            base_dir=output_goodimages_folder,
            pdf_name=pdf_name,
            output_file=os.path.join(
                output_base_extract_folder, f"{pdf_name}_descriptions.json"
            ),
        )

        doc_id = str(uuid.uuid4())

        document_info = {
            "doc_id": doc_id,
            "text_content": text_content.to_dict("records"),
            "image_descriptions": image_descriptions,
        }

        new_documents[pdf_name] = document_info

    return new_documents


def check_existing_embeddings(vectorstore: Chroma) -> None:
    """
    Print information about existing embeddings in the vectorstore.

    Parameters
    ----------
    vectorstore : Chroma
        The vectorstore to check embeddings from.
    """
    existing_docs = vectorstore.get()

    print(f"Total documents in vectorstore: {len(existing_docs['ids'])}")
    print("Existing document companies:")
    for metadata in existing_docs["metadatas"]:
        print(f"- {metadata.get('company', 'Unknown')}")
