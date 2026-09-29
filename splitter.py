from langchain_text_splitters import RecursiveCharacterTextSplitter

from loader import load_documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    documents = load_documents()

    chunks = split_documents(documents)

    print("Number of original documents:", len(documents))
    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks):

        print("\n" + "=" * 60)
        print(f"CHUNK {i + 1}")
        print("=" * 60)

        print(chunk.page_content)

        print("\nMetadata:")
        print(chunk.metadata)