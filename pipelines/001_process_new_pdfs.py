import sys

sys.path.append("..")

from function.retriever import (
    initialize_retriever,
    update_retriever,
    upload_retriever_to_cloud,
)


def main():
    retriever = initialize_retriever()
    update_retriever(retriever)
    # upload_retriever_to_cloud()


if __name__ == "__main__":
    main()
