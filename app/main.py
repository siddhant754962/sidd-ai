from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ingestion import ingest_youtube_video
from app.rag_chain import ask_question


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="VideoMind AI",
    description="Chat with YouTube videos using RAG",
    version="1.0.0"
)


# =========================================================
# REQUEST MODELS
# =========================================================

class IngestRequest(BaseModel):
    url: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


# =========================================================
# YOUTUBE URL VALIDATION
# =========================================================

def is_youtube_url(url: str):

    parsed = urlparse(url)
    hostname = parsed.netloc.lower()

    return (
        hostname == "youtube.com"
        or hostname.endswith(".youtube.com")
        or hostname == "youtu.be"
        or hostname.endswith(".youtu.be")
    )


# =========================================================
# API HOME
# =========================================================

@app.get("/api")
def api_home():

    return {
        "success": True,
        "message": "VideoMind AI API is running"
    }


# =========================================================
# INGEST VIDEO
# =========================================================

@app.post("/ingest")
def ingest_video(request: IngestRequest):

    if not is_youtube_url(request.url):

        raise HTTPException(
            status_code=400,
            detail="Please enter a valid YouTube URL."
        )

    try:

        result = ingest_youtube_video(request.url)

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask")
def ask_video_question(request: AskRequest):

    try:

        result = ask_question(
            video_id=request.video_id,
            question=request.question
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(e)}"
        )


# =========================================================
# FRONTEND
# =========================================================

app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend"
)