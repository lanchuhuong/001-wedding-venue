import os
import sys
import uuid
from tempfile import TemporaryDirectory

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from function.image import (  # noqa: E402
    get_all_venue_names_on_cloud,
    process_images,
)
from function.retriever import (  # noqa: E402
    PERSIST_DIRECTORY,
    add_documents_to_retriever,
    initialize_retriever,
    load_venue_metadata,
    upload_retriever_to_cloud,
)

print("Loading venue metadata...")
venue_metadata = load_venue_metadata()

venues = get_all_venue_names_on_cloud()
venue_infos = {}
retriever = initialize_retriever()
for venue in venues:
    print(f"Processing {venue}...")

    with TemporaryDirectory() as temp_output_dir:
        image_descriptions = process_images(venue, temp_output_dir, retriever)

    doc_id = str(uuid.uuid4())
    venue_info = venue_metadata.get(venue, {})
    document_info = {
        "doc_id": doc_id,
        "text_content": "",
        "image_descriptions": image_descriptions,
        "metadata": venue_info,
    }

    venue_infos[venue] = document_info

retriever = initialize_retriever()
add_documents_to_retriever(venue_infos, retriever, venue_metadata)
retriever.vectorstore.save_local(PERSIST_DIRECTORY)
upload_retriever_to_cloud()
