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


def download_audio(url):
    print("downloading the file")
    output_path = os.path.join(os.getcwd(), "audio.%(ext)s")
    print("Downloading audio to:", output_path)
    options = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        # No postprocessors = no ffmpeg needed
        # 'postprocessors': [...],  <-- removed
        # 'ffmpeg_location': '...',  <-- removed
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    print("download complete")
        
def merge_segments(segments, max_tokens=100):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = len(seg['text'].split())
        if current_tokens + seg_tokens <= max_tokens:
            current_chunk.append(seg)
            current_tokens += seg_tokens
        else:
            # Merge current_chunk
            merged_text = " ".join([s['text'].strip() for s in current_chunk])
            start_time = current_chunk[0]['start']
            end_time = current_chunk[-1]['end']
            chunks.append({
                "text": merged_text,
                "start": start_time,
                "end": end_time
            })
            current_chunk = [seg]
            current_tokens = seg_tokens

    # Add last chunk
    if current_chunk:
        merged_text = " ".join([s['text'].strip() for s in current_chunk])
        start_time = current_chunk[0]['start']
        end_time = current_chunk[-1]['end']
        chunks.append({
            "text": merged_text,
            "start": start_time,
            "end": end_time
        })

    return chunks

def create_chunks(text):
  documents = [
      Document(
          page_content=segment['text'].strip(),
          metadata = {
              'start': segment['start'],
              'end': segment['end'],
              'timestamp': f"{segment['start']:.2f} - {segment['end']:.2f}",

          }
      )
      for segment in text]

  chunks = RecursiveCharacterTextSplitter(
      chunk_size = 1000, chunk_overlap = 200
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
    download_audio(url)
    audio_path = os.path.join(os.getcwd(), "audio.webm")
    print(f"Looking for audio at: {audio_path}")
    
    model = whisper.load_model("base")
    
    result = model.transcribe(audio_path)
    # os.remove(audio_path)
    document = merge_segments(result['segments'])
    
    chunks = create_chunks(document)
    
    faiss_db = FAISS_DB(chunks)
    bm25_db, chunk_texts = BM_DB(chunks)
    
    os.remove(audio_path)
    return faiss_db, bm25_db, chunk_texts, chunks
    
