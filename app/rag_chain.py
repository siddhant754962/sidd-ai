import os

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.retriever import get_retriever


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# --------------------------------------------------
# LLM
# --------------------------------------------------

def get_llm():

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        huggingfacehub_api_token=HF_TOKEN,
        task="conversational",
        max_new_tokens=256,
        temperature=0.2,
    )

    llm = ChatHuggingFace(
        llm=llm_endpoint
    )

    return llm


# --------------------------------------------------
# FORMAT DOCUMENTS
# --------------------------------------------------

def format_documents(documents):

    formatted_documents = []

    for document in documents:

        start_time = document.metadata.get(
            "start_time",
            0
        )

        end_time = document.metadata.get(
            "end_time",
            0
        )

        formatted_text = f"""
[VIDEO TIMESTAMP: {start_time:.2f}s - {end_time:.2f}s]

{document.page_content}
"""

        formatted_documents.append(formatted_text)

    return "\n\n".join(formatted_documents)


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are VideoMind AI, an AI assistant that answers
questions about educational YouTube videos.

Answer the question using ONLY the provided context.

Each context section contains a video timestamp.

If the answer is not present in the context,
say that you do not have enough information.

Do not use outside knowledge.

Keep the answer clear and simple.

Do not invent timestamps.

Context:

{context}

Question:

{question}

Answer:
"""
)


# --------------------------------------------------
# YOUTUBE TIMESTAMP URL
# --------------------------------------------------

def create_youtube_timestamp_url(
    video_id,
    start_time
):

    start_time = int(start_time)

    return (
        f"https://www.youtube.com/watch?v={video_id}"
        f"&t={start_time}s"
    )


# --------------------------------------------------
# FORMAT SOURCES
# --------------------------------------------------

def format_sources(video_id, documents):

    sources = []

    for document in documents:

        start_time = document.metadata.get(
            "start_time",
            0
        )

        end_time = document.metadata.get(
            "end_time",
            0
        )

        source = {
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "url": create_youtube_timestamp_url(
                video_id,
                start_time
            ),
            "text": document.page_content
        }

        sources.append(source)

    return sources


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

def ask_question(video_id, question):

    # Retrieve documents ONCE
    retriever = get_retriever(
        video_id=video_id,
        k=3
    )

    source_documents = retriever.invoke(question)

    # Timestamp-aware context
    context = format_documents(
        source_documents
    )

    # LLM
    llm = get_llm()

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # Generate answer
    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # Generate clickable sources
    sources = format_sources(
        video_id,
        source_documents
    )

    return {
        "answer": answer,
        "sources": sources
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    video_id = input(
        "Enter YouTube video ID: "
    )

    question = input(
        "Enter your question: "
    )

    result = ask_question(
        video_id,
        question
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    print("=" * 60)

    for i, source in enumerate(
        result["sources"]
    ):

        print(f"\nSOURCE {i + 1}")

        print(
            f"Timestamp: "
            f"{source['start_time']:.2f}s"
            f" → "
            f"{source['end_time']:.2f}s"
        )

        print("\nYouTube URL:")
        print(source["url"])

        print("\nContent:")
        print(source["text"])