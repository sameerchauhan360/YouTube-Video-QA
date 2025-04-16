import streamlit as st
from preprocessing import get_documents  # Assuming improved version with error handling
from retriever import hybrid_ranked_chunks, prepare_context  # Assuming improved version
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser  # To get string output
from dotenv import load_dotenv
import logging  # For better logging
import os
import time

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


def get_api_key_from_env():
   with open(".env", "r") as file:
       for line in file:
           if line.startswith("OPENAI_API_KEY"):
               # Remove any leading/trailing spaces and get the value after the equal sign
               return line.split("=")[1].strip()
   return None

api_key = get_api_key_from_env()
# api_key = st.secrets["api_keys"]["openai"]
os.environ["OPENAI_API_KEY"] = api_key

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- LLM and Prompt Setup ---
try:
    llm = ChatOpenAI(model="gpt-4o-mini")
    logger.info("ChatOpenAI model loaded successfully.")
except Exception as e:
    logger.exception("Failed to initialize LLM.")
    st.error(
        f"Fatal Error: Failed to initialize the Language Model. Please check API keys and configuration. Error: {e}"
    )
    # Stop the app if LLM fails to load, as it's critical
    st.stop()

# Define the prompt template
prompt_template = """
You are an expert assistant providing accurate, concise, and context-specific answers based solely on the provided transcript context.

  ---

  ### Context:
  {context}

  ---
  ### Chat History:
  {chat_history}
  
  ---
  ### Question:
  {input}

  ---

  ### 🧠 Instructions:
  - **Only use the provided context** to answer the question.
  - If the answer cannot be determined from the context, clearly respond with:
    > "The context does not contain information related to this query."
  - **Do not explain** your reasoning; provide only the final answer.
  - Combine and paraphrase relevant information into a **natural and complete response**.
  - If applicable, format formulas using **Markdown/LaTeX-style** (e.g., `E = mc^2`).
  - **Cite the source using the full timestamp range** that covers the answer.

  ---

  ### ✅ Answer:
  [Your final answer here]

  ---

  ### ⏱️ Source:
  - **Timestamp Range:** [e.g., 132.12 - 154.56, 205.16 - 235.16]
  """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "input"],
)

# Using LCEL (LangChain Expression Language) pipe for the chain
# This chains the prompt, the LLM, and an output parser to get a string result
try:
    llm_chain = prompt | llm | StrOutputParser()
    logger.info("LLM chain created successfully.")
except Exception as e:
    logger.exception("Failed to create LLM chain.")
    st.error(f"Fatal Error: Failed to create the LLM processing chain. Error: {e}")
    st.stop()


# --- Helper Function for History Formatting ---
def format_history(history: list[dict]) -> str:
    """Formats chat history list into a readable string for the LLM."""
    if not history:
        return "No previous conversation history."
    buffer = []
    for chat in history:
        role = chat.get("role", "unknown").capitalize()
        message = chat.get("message", "")
        buffer.append(f"{role}: {message}")
    return "\n".join(buffer)


def get_summary(text, metadata):
    summary_prompt = """
    You are given a transcript and metadata of a YouTube video. Summarize the overall content of the video in a short, engaging paragraph that can be used as an introduction. 

    Focus on the main topics, purpose, and what the viewer can expect to learn or experience. Avoid overly technical language unless necessary, and keep the tone concise and informative.
    After the summary ends, formate this line by passing the thumbnail and webpage url which will be shown as the end of the summary.
    [![Video Thumbnail]({{thumbnail_url}})]({{webpage_url_url}})"
    
    Transcript:
    {text}
    
    Metadata:
    {metadata}
    
    """

    summary_prompt_template = PromptTemplate(
        template=summary_prompt, input_variables=["text", "metadata"]
    )

    summary = llm.invoke(summary_prompt_template.format(text=text, metadata=metadata))
    return summary.content


# --- Streamlit App ---

# Set page configuration (do this only once at the top)
st.set_page_config(page_title="YouTube Video Q&A", layout="wide")
st.title("❓ YouTube Video Q&A")

# Initialize session state more robustly
default_state = {
    "faiss_db": None,
    "bm25_db": None,
    "chunk_texts": None,  # Removed - redundant as chunks holds the Document objects
    "chunks": None,
    "history": [],
    "processed_url": None,  # Keep track of which URL is processed
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Sidebar for Video Processing ---
with st.sidebar:
    st.header("🎬 Video Processing")
    yt_url = st.text_input(
        "Enter YouTube video URL:",
        key="yt_url_input",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if st.button("Process Video", key="process_button"):
        if yt_url and yt_url.strip():
            # Check if it's a new URL
            if yt_url != st.session_state.processed_url:
                logger.info(f"Processing new URL: {yt_url}")
                # Clear previous state for the new video
                for key in default_state:
                    if key != "processed_url":  # Keep processed_url itself
                        st.session_state[key] = default_state[key]
                st.session_state.processed_url = (
                    None  # Reset processed URL until success
                )

                # Show spinner during processing
                with st.spinner(
                    "Processing video... (Downloading, Transcribing, Indexing) This may take a few minutes."
                ):
                    try:
                        # Call the preprocessing function (ensure it handles its own errors and returns None on failure)
                        faiss_db, bm25_db, chunks_texts, chunks, text, metadata = (
                            get_documents(yt_url)
                        )  # Assuming 3rd return is redundant chunk_texts
                        summary = get_summary(text, metadata)
                        # Check if processing was successful
                        if faiss_db and bm25_db and chunks:
                            st.session_state.faiss_db = faiss_db
                            st.session_state.bm25_db = bm25_db
                            st.session_state.chunks_texts = chunks_texts
                            st.session_state.chunks = chunks
                            st.session_state.history = (
                                []
                            )  # Reset history for the new video
                            st.session_state.processed_url = (
                                yt_url  # Store the successfully processed URL
                            )
                            st.session_state.summary = summary
                            st.success("✅ Video processed successfully!")
                            logger.info(
                                f"Successfully processed and indexed URL: {yt_url}"
                            )
                            # Use st.rerun() to update the main page state immediately
                            st.rerun()
                        else:
                            st.error(
                                "⚠️ Failed to process video. Please check the URL or logs."
                            )
                            logger.error(
                                f"get_documents returned None or incomplete data for URL: {yt_url}"
                            )
                            # Ensure state reflects failure
                            st.session_state.processed_url = None

                    except Exception as e:
                        st.error(
                            f"❌ An unexpected error occurred during processing: {e}"
                        )
                        logger.exception(f"Error processing URL {yt_url}:")
                        # Ensure state reflects failure
                        st.session_state.processed_url = None

            elif yt_url == st.session_state.processed_url:
                st.info("ℹ️ This URL has already been processed and is ready for Q&A.")
        else:
            st.warning("⚠️ Please enter a valid YouTube URL.")

    # Display the currently processed URL
    if st.session_state.processed_url:
        st.markdown("---")
        st.markdown(f"**✨ Ready for Q&A on:**\n`{st.session_state.processed_url}`")


# --- Main Chat Area ---
st.header("💬 Chat with the Video")
if "summary" in st.session_state:
    with st.expander("📌 Video Summary", expanded=True):
        st.markdown(st.session_state.summary)

# Only show chat interface if a video has been processed
if not st.session_state.faiss_db or not st.session_state.chunks:
    st.info(
        "👋 Welcome! Please process a YouTube video using the sidebar to start chatting."
    )
else:
    # Display existing chat messages from history
    for chat in st.session_state.history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])

    # Get user input via chat input box
    user_input = st.chat_input("Ask a question about the video content...")

    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        # Add user message to history
        st.session_state.history.append({"role": "user", "message": user_input})

        # Retrieve relevant chunks using hybrid search
        with st.spinner("🔍 Searching relevant video segments..."):
            try:
                retrieved_chunks = hybrid_ranked_chunks(
                    user_input,
                    st.session_state.faiss_db,
                    st.session_state.bm25_db,
                    st.session_state.chunks_texts,  # Pass the actual list of Document objects
                    st.session_state.chunks,  # Pass the actual list of Document objects
                    top_k=5,  # Or make this configurable
                )
                context = prepare_context(retrieved_chunks)
                logger.info(
                    f"Retrieved {len(retrieved_chunks)} chunks for query: '{user_input}'"
                )

                # Optional: Display retrieved context for debugging
                # with st.expander("View Retrieved Context"):
                #    st.text(context if context else "No context retrieved.")

            except Exception as e:
                st.error(f"❌ Error during retrieval: {e}")
                logger.exception(f"Error retrieving chunks for query: {user_input}")
                context = None  # Ensure context is None if retrieval fails

        # Generate response only if context retrieval was successful (or handle differently)
        if context is not None:  # Proceed only if retrieval worked
            with st.spinner("🧠 Thinking..."):
                try:
                    # Format history *before* the current user query for the LLM context
                    formatted_history = format_history(st.session_state.history[:-1])

                    # Invoke the LLM chain with the correct input variable names
                    # print(context)
                    response = llm_chain.invoke(
                        {
                            "context": context,
                            "chat_history": formatted_history,
                            "input": user_input,
                        }
                    )
                    # print(response)
                    logger.info("LLM response generated successfully.")
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = ""
                        for char in response:
                            full_response += char
                            message_placeholder.markdown(
                                full_response + "|"
                            )  # add a cursor
                            time.sleep(0.02)
                        message_placeholder.markdown(full_response)

                    # Add assistant response to history
                    st.session_state.history.append(
                        {"role": "assistant", "message": response}
                    )

                except Exception as e:
                    st.error(f"❌ Error generating response from AI: {e}")
                    logger.exception("Error during LLM invocation:")
                    # Optionally add an error message to the chat history for the user
                    error_message = (
                        "[Sorry, I encountered an error trying to generate a response.]"
                    )
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    st.session_state.history.append(
                        {"role": "assistant", "message": error_message}
                    )

        # Rerun to clear the input box after processing the message (optional, can feel smoother)
        # st.rerun()
