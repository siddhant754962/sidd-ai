from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    return embeddings


if __name__ == "__main__":

    embeddings = get_embedding_model()

    text = "Artificial intelligence is changing the way people learn."

    vector = embeddings.embed_query(text)

    print("Embedding created successfully!")
    print("Model:", MODEL_NAME)
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])