import whisper
from whisper import load_audio
import yt_dlp
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_video_metadata(url):
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "id": info.get("id", ""),
            "title": info.get("title", ""),
            "description": info.get("description", ""),
            "upload_date": info.get("upload_date", ""),
            "uploader": info.get("uploader", ""),
            "duration": info.get("duration", 0),
            "view_count": info.get("view_count", 0),
            "like_count": info.get("like_count", 0),
            'webpage_url': info.get("webpage_url", ''),
            'thumbnail': info.get('thumbnail', '')
        }


# def download_audio(url):
#     print("downloading the file")
#     output_path = os.path.join(os.getcwd(), "audio.%(ext)s")
#     print("Downloading audio to:", output_path)
#     options = {
#         "format": "bestaudio/best",
#         "outtmpl": output_path,
#         "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
#     }

#     with yt_dlp.YoutubeDL(options) as ydl:
#         ydl.download([url])
#     print("download complete")
    
def download_audio(url, id):
    import os
    import yt_dlp

    print("Downloading audio from:", url)
    output_path = os.path.join(os.getcwd(), f"{id}.webm")
    
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "nocheckcertificate": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
        print("Download complete.")
    except Exception as e:
        print("❌ Download failed:", str(e))


def merge_segments(segments, max_tokens=100):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = len(seg["text"].split())
        if current_tokens + seg_tokens <= max_tokens:
            current_chunk.append(seg)
            current_tokens += seg_tokens
        else:
            # Merge current_chunk
            merged_text = " ".join([s["text"].strip() for s in current_chunk])
            start_time = current_chunk[0]["start"]
            end_time = current_chunk[-1]["end"]
            chunks.append({"text": merged_text, "start": start_time, "end": end_time})
            current_chunk = [seg]
            current_tokens = seg_tokens

    # Add last chunk
    if current_chunk:
        merged_text = " ".join([s["text"].strip() for s in current_chunk])
        start_time = current_chunk[0]["start"]
        end_time = current_chunk[-1]["end"]
        chunks.append({"text": merged_text, "start": start_time, "end": end_time})

    return chunks


def create_chunks(text, video_metadata):
    documents = [
        Document(
            page_content=segment["text"].strip(),
            metadata={
                "start": segment["start"],
                "end": segment["end"],
                "timestamp": f"{segment['start']:.2f} - {segment['end']:.2f}",
                **(video_metadata or {}),
            },
        )
        for segment in text
    ]

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(documents)

    return chunks


def FAISS_DB(chunks):
    """Creates a FAISS vector database from document chunks."""
    faiss_db = FAISS.from_documents(chunks, embeddings)
    return faiss_db


def BM_DB(chunks):
    """Creates a BM25 database from document chunks."""
    tokenized_chunks = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, chunks


def get_documents(url):
    metadata = get_video_metadata(url)  # Get video metadata

    download_audio(url, metadata['id'])
    audio_path = os.path.join(os.getcwd(), f"{metadata['id']}.webm")
    print(f"Looking for audio at: {audio_path}")

    model = whisper.load_model("base")

    result = model.transcribe(audio_path)
    print(result)
    document = merge_segments(result["segments"])

    chunks = create_chunks(document, video_metadata=metadata)

    faiss_db = FAISS_DB(chunks)
    bm25_db, chunk_texts = BM_DB(chunks)

    os.remove(audio_path)

    return faiss_db, bm25_db, chunk_texts, chunks, result["text"], metadata
