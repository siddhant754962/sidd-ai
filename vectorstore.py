from langchain_chroma import Chroma

from app.embeddings import get_embedding_model


COLLECTION_NAME = "videomind_youtube"

VECTORSTORE_PATH = "data/vectorstore"


# --------------------------------------------------
# GET VECTORSTORE
# --------------------------------------------------

def get_vectorstore():

    embeddings = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    return vectorstore


# --------------------------------------------------
# CHECK VIDEO
# --------------------------------------------------

def video_already_indexed(video_id):

    vectorstore = get_vectorstore()

    results = vectorstore.get(
        where={
            "video_id": video_id
        },
        limit=1
    )

    return len(results["ids"]) > 0


# --------------------------------------------------
# CREATE / ADD DOCUMENTS
# --------------------------------------------------

def create_vectorstore(
    chunks,
    video_id
):

    vectorstore = get_vectorstore()

    ids = []

    for index, chunk in enumerate(chunks):

        chunk_id = (
            f"{video_id}_chunk_{index}"
        )

        ids.append(chunk_id)

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    return vectorstore