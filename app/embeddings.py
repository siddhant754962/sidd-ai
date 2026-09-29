import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedding_model():
    """
    Return a Hugging Face hosted embedding client.

    The embedding model runs remotely,
    so Render does not need to load
    Sentence Transformers / PyTorch.
    """

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is not configured."
        )

    client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN
    )

    return client


def embed_text(text: str):

    client = get_embedding_model()

    embedding = client.feature_extraction(
        text,
        model=EMBEDDING_MODEL
    )

    return embedding


if __name__ == "__main__":

    text = (
        "Artificial intelligence is changing "
        "the way people learn."
    )

    vector = embed_text(text)

    print("Embedding created successfully!")

    print(
        "Model:",
        EMBEDDING_MODEL
    )

    print(
        "Vector dimensions:",
        len(vector)
    )

    print(
        "First 10 values:",
        vector[:10]
    )
