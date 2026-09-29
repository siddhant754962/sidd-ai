from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document


# --------------------------------------------------
# EXTRACT VIDEO ID
# --------------------------------------------------

def get_video_id(url: str) -> str:

    parsed_url = urlparse(url)

    hostname = parsed_url.hostname.lower() if parsed_url.hostname else ""

    if hostname in ["youtube.com", "www.youtube.com"]:

        query = parse_qs(parsed_url.query)

        if "v" in query:
            return query["v"][0]

    if hostname in ["youtu.be", "www.youtu.be"]:

        return parsed_url.path.strip("/")

    raise ValueError("Invalid YouTube URL")


# --------------------------------------------------
# GET TRANSCRIPT
# --------------------------------------------------

def get_transcript(video_id: str):

    api = YouTubeTranscriptApi()

    return api.fetch(video_id)


# --------------------------------------------------
# CREATE CHUNKS
# --------------------------------------------------

def transcript_to_chunks(
    transcript,
    video_id: str,
    chunk_duration: int = 75,
    overlap: int = 15,
    **kwargs
):
    """
    Create timestamp-aware overlapping chunks.

    Default:
        chunk_duration = 75 seconds
        overlap = 15 seconds

    **kwargs is intentionally accepted so an old
    chunk_size argument does not crash the application.
    """

    # ----------------------------------------------
    # Backward compatibility
    # ----------------------------------------------

    if "chunk_size" in kwargs:

        print(
            "Warning: chunk_size is deprecated. "
            "Using chunk_duration=75 instead."
        )

        chunk_duration = 75

    chunks = []

    if not transcript:
        return chunks

    # ----------------------------------------------
    # Convert transcript segments
    # ----------------------------------------------

    segments = []

    for segment in transcript:

        text = segment.text.strip()

        start = float(segment.start)

        end = float(
            segment.start + segment.duration
        )

        if text:

            segments.append({
                "text": text,
                "start": start,
                "end": end
            })

    if not segments:
        return chunks

    # ----------------------------------------------
    # Build overlapping chunks
    # ----------------------------------------------

    current_segments = []

    chunk_start = segments[0]["start"]

    for segment in segments:

        current_segments.append(segment)

        chunk_end = segment["end"]

        # ------------------------------------------
        # Target duration reached
        # ------------------------------------------

        if chunk_end - chunk_start >= chunk_duration:

            combined_text = " ".join(
                item["text"]
                for item in current_segments
            )

            chunks.append(
                Document(
                    page_content=combined_text,
                    metadata={
                        "source": "youtube",
                        "video_id": video_id,
                        "start_time": chunk_start,
                        "end_time": chunk_end
                    }
                )
            )

            # --------------------------------------
            # Keep last 15 seconds
            # --------------------------------------

            overlap_start = chunk_end - overlap

            current_segments = [
                item
                for item in current_segments
                if item["end"] > overlap_start
            ]

            if current_segments:

                chunk_start = current_segments[0]["start"]

            else:

                chunk_start = chunk_end

    # ----------------------------------------------
    # Remaining chunk
    # ----------------------------------------------

    if current_segments:

        combined_text = " ".join(
            item["text"]
            for item in current_segments
        )

        chunks.append(
            Document(
                page_content=combined_text,
                metadata={
                    "source": "youtube",
                    "video_id": video_id,
                    "start_time": current_segments[0]["start"],
                    "end_time": current_segments[-1]["end"]
                }
            )
        )

    return chunks


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    url = input("Enter YouTube URL: ")

    video_id = get_video_id(url)

    print("\nVideo ID:")
    print(video_id)

    transcript = get_transcript(video_id)

    print("\nTranscript segments:")
    print(len(transcript))

    chunks = transcript_to_chunks(
        transcript,
        video_id,
        chunk_duration=75,
        overlap=15
    )

    print("\nChunks created:")
    print(len(chunks))

    for i, chunk in enumerate(chunks[:5]):

        print("\n" + "=" * 70)

        print(f"CHUNK {i + 1}")

        print(
            f"Timestamp: "
            f"{chunk.metadata['start_time']:.2f}s"
            f" → "
            f"{chunk.metadata['end_time']:.2f}s"
        )

        print("\nText:")

        print(chunk.page_content)

        print("\nMetadata:")

        print(chunk.metadata)