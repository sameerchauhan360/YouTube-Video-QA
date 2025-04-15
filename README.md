# YouTube Video Q&A ❓🎬

This project allows you to "chat" with a YouTube video. You provide a YouTube video URL, the application processes it by downloading the audio, transcribing it, and indexing the content. You can then ask questions about the video, and an AI assistant (powered by OpenAI's GPT models) will answer based *only* on the information found in the video's transcript.

The system uses a hybrid search approach (combining semantic search with FAISS and keyword search with BM25) to find the most relevant parts of the transcript to answer your questions accurately.

## ✨ Features

*   **YouTube Video Processing:** Downloads audio from a given YouTube URL.
*   **Automatic Transcription:** Uses OpenAI's Whisper model to transcribe the video audio.
*   **Content Indexing:** Chunks the transcript and creates searchable indexes using:
    *   **FAISS:** For efficient semantic similarity search.
    *   **BM25:** For keyword-based relevance ranking.
*   **Hybrid Search:** Combines FAISS and BM25 scores to retrieve the most relevant transcript chunks for a given query.
*   **Context-Aware Q&A:** Uses a Large Language Model (LLM - specifically GPT-4o-mini via Langchain) to generate answers based *strictly* on the retrieved transcript context.
*   **Video Summarization:** Generates a brief summary of the video content after processing.
*   **Chat Interface:** Provides a user-friendly web interface built with Streamlit to interact with the processed video.
*   **Source Citation:** Answers include the timestamp range(s) from the video transcript where the information was found.
*   **Chat History:** Remembers the conversation flow within a session.

## 🛠️ Technology Stack

*   **Backend:** Python
*   **Web Framework:** Streamlit
*   **YouTube Downloader:** `yt-dlp`
*   **Transcription:** `openai-whisper`
*   **LLM Orchestration:** `langchain`, `langchain-openai`
*   **Embeddings:** `sentence-transformers` (via `langchain-huggingface`)
*   **Vector Store:** `faiss-cpu` (or `faiss-gpu`)
*   **Keyword Search:** `rank_bm25`
*   **LLM:** OpenAI API (GPT-4o-mini)
*   **Environment Variables:** `python-dotenv`

## ⚙️ Setup and Installation
1.  **Prerequisite: Install FFmpeg:**
    *   `whisper` requires `ffmpeg` to be installed on your system for audio processing.
    *   **On macOS:** `brew install ffmpeg`
    *   **On Windows:** Download from the official FFmpeg website and add it to your system's PATH.
    *   **On Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/sameerchauhan360/YouTube-Video-QA.git
    cd YouTube-Video-QA
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    
    # On Windows
    .\venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Create a `requirements.txt` file:**
    Based on your imports, your `requirements.txt` file should look something like this (you might need to adjust versions):
    ```txt
    streamlit
    openai-whisper
    yt-dlp
    langchain
    langchain-openai
    langchain-huggingface
    faiss-cpu # or faiss-gpu if you have CUDA setup
    rank_bm25
    sentence-transformers
    scikit-learn # For MinMaxScaler
    python-dotenv
    openai # Required by langchain-openai
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Set up Environment Variables:**
    *   Create a file named `.env` in the root directory of the project.
    *   Add your OpenAI API key to this file:
        ```env
        OPENAI_API_KEY=your_openai_api_key_here
        ```
    *   *Note:* The `app.py` currently reads the key directly from the file. Ensure this method is secure for your deployment environment. Using `load_dotenv()` and `os.getenv("OPENAI_API_KEY")` within the app is generally preferred after loading.

## ▶️ Usage

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your Web Browser:** Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Process a Video:**
    *   Paste the URL of the YouTube video you want to query into the text input box in the sidebar.
    *   Click the "Process Video" button.
    *   Wait for the application to download the audio, transcribe it, and create the search indexes. This might take a few minutes depending on the video length and your machine's performance. You'll see status messages and spinners indicating progress.

4.  **Ask Questions:**
    *   Once the video is processed successfully, a summary will appear, and the chat interface will be ready.
    *   Type your questions about the video content into the chat input box at the bottom and press Enter.
    *   The assistant will retrieve relevant information from the transcript and provide an answer along with the source timestamps.

## 📝 Notes

*   **Transcription Model:** The `preprocessing.py` script currently uses the `base` Whisper model. You can change this (e.g., to `small`, `medium`) for potentially higher accuracy at the cost of processing time and computational resources.
*   **Error Handling:** While some basic error handling is included, further improvements can be made for robustness, especially around network issues, invalid URLs, and API failures.
*   **Resource Usage:** Transcription and embedding generation can be resource-intensive. Processing long videos may require significant time and memory.

## 🚀 Potential Improvements

*   Add caching for processed videos to avoid reprocessing the same URL.
*   Allow selection of different Whisper models via the UI.
*   Implement more robust error handling and user feedback.
*   Optimize the hybrid search weighting or allow user adjustment.
*   Support for video formats other than YouTube (e.g., local files).
*   Deploy as a persistent web service.

---

*Feel free to add a License section if applicable.*
