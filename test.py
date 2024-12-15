import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import Client, transfer_manager

load_dotenv()

BUCKET_NAME = "wedding-venues-001"


def list_files(filter=None):
    if filter is not None:
        filter = re.compile(filter, re.IGNORECASE)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs if filter is None or filter.search(blob.name)]


def download_file(source_blob_name: str, destination_file_name: str):
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    return destination_file_name


def download_files(files: list[str]):
    client = Client()
    bucket = client.bucket(BUCKET_NAME)

    downloads = [(bucket.blob(file_name), file_name) for file_name in files]

    for _, dest_path in downloads:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    results = transfer_manager.download_many(
        downloads,
        max_workers=10,
    )

    return results


def upload_file(source_file_path: str, destination_blob_name: str | None = None) -> str:
    """
    Upload a single file to Google Cloud Storage.

    Parameters
    ----------
    source_file_path : str
        Local path to the file to upload
    destination_blob_name : str, optional
        Destination path in the bucket. If None, uses the source filename

    Returns
    -------
    str
        The public URL of the uploaded file
    """
    if destination_blob_name is None:
        destination_blob_name = Path(source_file_path).name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)

    return blob.public_url


def upload_files(file_pairs: list[tuple[str, str]]) -> list[str]:
    """
    Upload multiple files to Google Cloud Storage in parallel.

    Parameters
    ----------
    file_pairs : list[tuple[str, str]]
        List of (source_path, destination_path) tuples

    Returns
    -------
    list[str]
        List of public URLs for the uploaded files
    """
    client = Client()
    bucket = client.bucket(BUCKET_NAME)

    # Create list of (source_file, destination_blob) tuples
    uploads = [
        (file_path, bucket.blob(dest_path)) for file_path, dest_path in file_pairs
    ]

    # Upload files in parallel
    results = transfer_manager.upload_many(
        uploads,
        max_workers=10,
    )

    # Return public URLs of uploaded files
    return [blob.public_url for _, blob in uploads]


def upload_directory(local_directory: str, bucket_prefix: str = "") -> list[str]:
    """
    Upload an entire directory and its contents recursively.

    Parameters
    ----------
    local_directory : str
        Path to local directory to upload
    bucket_prefix : str, optional
        Prefix to add to all files in the bucket

    Returns
    -------
    list[str]
        List of public URLs for all uploaded files
    """
    local_dir = Path(local_directory)

    # Get all files in directory and subdirectories
    all_files = [
        (str(path), os.path.join(bucket_prefix, path.relative_to(local_dir).as_posix()))
        for path in local_dir.rglob("*")
        if path.is_file()
    ]

    return upload_files(all_files)


if __name__ == "__main__":
    upload_file("app.py", "app6.py")
    list_files("app")
