import os
import re
import sys
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from function.cloud import (  # noqa: E402
    delete_file,
    download_file,
    download_files,
    list_files,
)
from function.process_image import (  # noqa: E402
    generate_image_descriptions,
    is_photo,
    load_is_photo_classifier,
)
from function.retriever import (  # noqa: E402
    PERSIST_DIRECTORY,
    add_documents_to_retriever,
    initialize_retriever,
    load_venue_metadata,
    upload_retriever_to_cloud,
)


def get_venue_images_from_cloud(venue):
    images = list_files(f"{venue}/figures/.*")
    return images


def get_all_venue_names_on_cloud():
    venue_paths = list_files(r"venues/.*")
    pattern = re.compile("venues/(.*)/.*.pdf")
    venue_names = [pattern.findall(path)[0] for path in venue_paths]
    return venue_names


def download_to_folder(files, destination_folder):
    download_files(files, [destination_folder + "/" + file for file in files])


def get_venue_images_from_receiver(venue):
    receiver = initialize_retriever()
    docs = receiver.vectorstore.docstore._dict.values()
    image_docs = filter(
        lambda doc: doc.metadata["type"] == "image"
        and doc.metadata["company"] == venue,
        docs,
    )
    image_paths = [doc.metadata["image_path"] for doc in image_docs]
    image_names = [os.path.basename(path) for path in image_paths]
    return image_names


venue_metadata = load_venue_metadata()
photo_classifier = load_is_photo_classifier()

venues = get_all_venue_names_on_cloud()

venue_infos = {}

for venue in venues:
    print(f"Processing {venue}...")
    cloud_images = get_venue_images_from_cloud(venue)
    receiver_images = get_venue_images_from_receiver(venue)

    with TemporaryDirectory() as temp_output_dir:
        if len(cloud_images) > 0:
            root = os.path.dirname(cloud_images[0])
            cloud_image_names = set([os.path.basename(image) for image in cloud_images])
            receiver_image_names = set(receiver_images)
            images_not_in_receiver = cloud_image_names - receiver_image_names
            print(f"Found {len(images_not_in_receiver)} images not in receiver:")
            print(f"  {images_not_in_receiver}")

        for image in images_not_in_receiver:
            image_on_disk = temp_output_dir + "/" + image
            print(f"Downloading {image}...")
            download_file(root + "/" + image, image_on_disk)
            if not is_photo(photo_classifier, image_on_disk):
                print(f"{image} not photo. Deleting {image}...")
                delete_file(root + "/" + image)
                Path(image_on_disk).unlink()

        print(f"Generating image descriptions for {venue}...")
        image_descriptions = generate_image_descriptions(
            base_dir=temp_output_dir,
            venue=venue,
        )

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
# upload_retriever_to_cloud()
