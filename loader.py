from pathlib import Path
from langchain_community.document_loaders import TextLoader


def load_documents():

    document_path = Path("data/documents/sample.txt")

    loader = TextLoader(
        str(document_path),
        encoding="utf-8"
    )

    documents = loader.load()

    return documents


if __name__ == "__main__":

    documents = load_documents()

    print("Number of documents:", len(documents))

    for document in documents:

        print("\n--- DOCUMENT ---")
        print("Content:")
        print(document.page_content[:500])

        print("\nMetadata:")
        print(document.metadata)