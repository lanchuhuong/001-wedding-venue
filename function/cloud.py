import os
import re
from pathlib import Path
from typing import List

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from dotenv import load_dotenv
from google.cloud import storage
from google.cloud.storage import transfer_manager
from google.oauth2 import service_account

load_dotenv()

try:
    # Initialize storage_client as None first
    storage_client = None

    # For local development and github actions
    if (not STREAMLIT_AVAILABLE) or os.path.exists(
        "turing-guard-444623-s7-2cd0a98f8177.json"
    ):
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


def list_files(filter=None):
    if filter is not None:
        filter = re.compile(filter, re.IGNORECASE)

    blobs = bucket.list_blobs()
    return [blob.name for blob in blobs if filter is None or filter.search(blob.name)]


def download_file(source_blob_name: str, destination_file_name: str):
    source_blob_name = Path(source_blob_name).as_posix()
    destination_file_name = Path(destination_file_name).as_posix()
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    return destination_file_name


def download_files(
    files: list[str], destination_files: list[str] | None = None, verbose=False
):
    if destination_files is None:
        destination_files = files

    downloads = [
        (bucket.blob(file_name), destination_file_name)
        for file_name, destination_file_name in zip(files, destination_files)
    ]

    for _, dest_path in downloads:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    results = transfer_manager.download_many(
        downloads,
        max_workers=10,
    )
    if verbose:
        for blob_name, dest_name in zip(files, destination_files):
            print(f"downloaded {blob_name} to {dest_name}")
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
    Upload an entire directory and its contents recursively, skipping files that already exist.

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

    all_files = [
        (str(path), os.path.join(bucket_prefix, path.relative_to(local_dir).as_posix()))
        for path in local_dir.rglob("*")
        if path.is_file()
    ]

    # Filter out files that already exist in the bucket
    files_to_upload = []
    for local_path, bucket_path in all_files:
        blob = bucket.blob(bucket_path.lstrip("/"))
        if not blob.exists():
            files_to_upload.append((local_path, bucket_path))
        else:
            print(f"Skipping existing file: {bucket_path}")

    if not files_to_upload:
        print("All files already exist in the bucket. Nothing to upload.")
        return []

    return upload_files(files_to_upload)


def delete_file(blob_name: str) -> bool:
    """
    Delete a file from Google Cloud Storage.

    Parameters
    ----------
    blob_name : str
        Path to the file in the bucket to delete

    Returns
    -------
    bool
        True if deletion was successful, False otherwise
    """
    try:
        blob = bucket.blob(blob_name)
        blob.delete()
        return True
    except Exception as e:
        print(f"Error deleting file {blob_name}: {str(e)}")
        return False


def download_directory(venue: str, target_dir: str, verbose: bool = False) -> List[str]:
    """
    Downloads the adobe_extracted directory for a specific venue from Google Cloud Storage
    to the current working directory.

    Parameters
    ----------
    venue : str
        Name of the venue folder to download
    target_dir : str
        ...

    Returns
    -------
    List[str]
        List of downloaded file paths
    """
    prefix = f"processed/adobe_extracted/{venue}/"

    print(f"Using prefix: {prefix}")
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        print(f"No files found for venue: {prefix}")
        return []

    blob_names = [blob.name for blob in blobs]
    if target_dir is None:
        target_dir = "."
    target_blob_names = [
        os.path.join(target_dir, blob_name.replace(prefix, ""))
        for blob_name in blob_names
    ]
    downloaded_files = []
    download_files(blob_names, target_blob_names, verbose=verbose)

    print(f"Successfully downloaded {len(downloaded_files)} files for venue {venue}")
    return downloaded_files
