import os
import re
from pathlib import Path

from function.cloud import (  # noqa: E402
    delete_file,
    download_file,
    list_files,
)
from function.process_image import (  # noqa: E402
    generate_image_descriptions,
    is_photo,
    load_is_photo_classifier,
)


def get_venue_images_from_cloud(venue):
    images = list_files(f"{venue}/figures/.*")
    return images


def get_all_venue_names_on_cloud():
    venue_paths = list_files(r"venues/.*")
    venue_paths = [path for path in venue_paths if path.endswith(".pdf")]
    pattern = re.compile("venues/(.*)/.*.pdf")
    venue_names = [pattern.findall(path)[0] for path in venue_paths]
    return venue_names


def get_venue_images_from_receiver(venue, receiver):
    docs = receiver.vectorstore.docstore._dict.values()
    image_docs = filter(
        lambda doc: doc.metadata["type"] == "image"
        and doc.metadata["company"] == venue,
        docs,
    )
    image_paths = [doc.metadata["image_path"] for doc in image_docs]
    image_names = [os.path.basename(path) for path in image_paths]
    return image_names


photo_classifier = load_is_photo_classifier()


def process_images(venue, temp_output_dir, receiver):
    cloud_images = get_venue_images_from_cloud(venue)
    receiver_images = get_venue_images_from_receiver(venue, receiver)
    images_not_in_receiver = []
    if len(cloud_images) > 0:
        root = os.path.dirname(cloud_images[0])
        if root.startswith("/"):
            root = root[1:]

        cloud_image_names = set([os.path.basename(image) for image in cloud_images])
        receiver_image_names = set(receiver_images)
        images_not_in_receiver = cloud_image_names - receiver_image_names
        print(f"Found {len(images_not_in_receiver)} images not in receiver:")
        print(f"  {images_not_in_receiver}")

    for image in images_not_in_receiver:
        image_on_disk = os.path.join(temp_output_dir, image)
        print(f"Downloading {image}...")
        try:
            download_file(os.path.join(root, image), image_on_disk)
            if not is_photo(photo_classifier, image_on_disk):
                print(f"{image} not photo. Deleting {image}...")
                delete_file(os.path.join(root, image))
                Path(image_on_disk).unlink()
        except Exception as e:
            print(f"Error processing file: {e}")

    print(f"Generating image descriptions for {venue}...")
    image_descriptions = generate_image_descriptions(
        base_dir=temp_output_dir,
        venue=venue,
    )
    return image_descriptions
