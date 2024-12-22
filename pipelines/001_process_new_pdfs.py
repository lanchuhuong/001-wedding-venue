import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from function.retriever import (  # noqa: E402
    initialize_retriever,
    load_venue_metadata,
    update_retriever,
    upload_retriever_to_cloud,
)


def main():
    retriever = initialize_retriever()
    venue_metadata = load_venue_metadata()
    update_retriever(retriever, venue_metadata)
    # upload_retriever_to_cloud()


if __name__ == "__main__":
    main()
