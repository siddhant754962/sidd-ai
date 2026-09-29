from langchain_chroma import Chroma
from app.embeddings import get_embedding_model

COLLECTION_NAME = "videomind_youtube"
VECTORSTORE_PATH = "data/vectorstore"


def get_vectorstore():

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    return vectorstore


def get_retriever(video_id, k=6):

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {
                "video_id": video_id
            }
        }
    )

    return retriever


if __name__ == "__main__":

    video_id = input("Enter YouTube video ID: ")

    question = input("Enter your question: ")

    retriever = get_retriever(
        video_id=video_id,
        k=3
    )

    documents = retriever.invoke(question)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, document in enumerate(documents):

        print(f"\n--- RESULT {i + 1} ---")

        print("\nContent:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)