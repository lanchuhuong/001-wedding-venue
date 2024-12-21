import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from function.pdf_loader import adobeLoader, extract_text_from_file_adobe
from function.process_image import generate_image_descriptions


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
    output_base_zip_path = Path("data/processed/adobe_result/")
    output_base_extract_folder = Path("data/processed/adobe_extracted/")
    output_good_images_folder = Path(os.getenv("OUTPUT_IMAGES_DIR", ""))

    output_good_images_folder.mkdir(exist_ok=True)

    new_documents: dict[str, dict[str, Any]] = {}

    for pdf_path in tqdm(pdf_paths):
        print(f"processing {pdf_path}")
        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
        output_zip_path = os.path.join(output_base_zip_path, pdf_name, "sdk.zip")
        output_zipextract_folder = os.path.join(output_base_extract_folder, pdf_name)
        client_id = os.getenv("ADOBE_CLIENT_ID")
        client_secret = os.getenv("ADOBE_CLIENT_SECRET")
        if not os.path.exists(
            os.path.join(output_zipextract_folder, "structuredData.json")
        ):
            print(f"loading {pdf_name} to adobe pdf services...")
            adobeLoader(
                pdf_path,
                output_zip_path=output_zip_path,
                client_id=client_id,
                client_secret=client_secret,
            )
        text_content = extract_text_from_file_adobe(
            output_zip_path, output_zipextract_folder
        )
        # df["company"] = pdf_name
        # text_content = (
        #     df.groupby("company")["text"].apply(lambda x: "\n".join(x)).reset_index()
        # )
        extracted_figure_folder = Path(output_zipextract_folder) / "figures"
        if not extracted_figure_folder.exists():
            image_descriptions = []
        else:
            image_descriptions = generate_image_descriptions(
                base_dir=extracted_figure_folder.as_posix(),
                pdf_name=pdf_name,
                output_file=os.path.join(
                    output_base_extract_folder, f"{pdf_name}_descriptions.json"
                ),
            )

        doc_id = str(uuid.uuid4())

        document_info = {
            "doc_id": doc_id,
            "text_content": text_content,
            "image_descriptions": image_descriptions,
        }

        new_documents[pdf_name] = document_info

    return new_documents
