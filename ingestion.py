from app.youtube_loader import (
    get_video_id,
    get_transcript,
    transcript_to_chunks
)

from app.vectorstore import (
    video_already_indexed,
    create_vectorstore
)


# --------------------------------------------------
# INGEST YOUTUBE VIDEO
# --------------------------------------------------

def ingest_youtube_video(url):

    # ----------------------------------------------
    # STEP 1 — Extract video ID
    # ----------------------------------------------

    video_id = get_video_id(url)

    print("\nVideo ID:")
    print(video_id)

    # ----------------------------------------------
    # STEP 2 — Check duplicate
    # ----------------------------------------------

    if video_already_indexed(video_id):

        return {
            "success": True,
            "message": "Video is already indexed.",
            "video_id": video_id,
            "chunks": 0,
            "already_exists": True
        }

    # ----------------------------------------------
    # STEP 3 — Get transcript
    # ----------------------------------------------

    transcript = get_transcript(video_id)

    print("\nTranscript segments:")
    print(len(transcript))

    if not transcript:

        raise ValueError(
            "No transcript was found for this video."
        )

    # ----------------------------------------------
    # STEP 4 — Create timestamp chunks
    # ----------------------------------------------

    chunks = transcript_to_chunks(
        transcript,
        video_id,
        chunk_duration=130,
        chunk_overlap=10
    )

    print("\nChunks created:")
    print(len(chunks))

    if not chunks:

        raise ValueError(
            "Could not create chunks from transcript."
        )

    # ----------------------------------------------
    # STEP 5 — Store in ChromaDB
    # ----------------------------------------------

    create_vectorstore(
        chunks,
        video_id
    )

    # ----------------------------------------------
    # STEP 6 — Return result
    # ----------------------------------------------

    return {
        "success": True,
        "message": "Video indexed successfully.",
        "video_id": video_id,
        "chunks": len(chunks),
        "already_exists": False
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    url = input("Enter YouTube URL: ")

    result = ingest_youtube_video(url)

    print("\n" + "=" * 60)
    print("INGESTION RESULT")
    print("=" * 60)

    print(result)