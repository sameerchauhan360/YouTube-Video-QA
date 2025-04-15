from collections import defaultdict
from typing import List
from sklearn.preprocessing import MinMaxScaler


def hybrid_ranked_chunks(
    query: str,
    faiss_db,
    bm25_db,
    chunk_texts: List,
    chunks: List,
    top_k: int = 5,
    w1=0.7,
    w2=0.3,
):
    tokenized_query = query.lower().split()

    # 1. FAISS retrieval with scores
    faiss_raw = faiss_db.similarity_search_with_score(query, k=top_k)
    faiss_docs = [doc for doc, _ in faiss_raw]
    faiss_scores = [score for _, score in faiss_raw]

    # Normalize FAISS scores (lower = better → invert)
    faiss_scaled = MinMaxScaler().fit_transform([[s] for s in faiss_scores])
    faiss_normalized = [1 - s[0] for s in faiss_scaled]  # higher is better

    # 2. BM25 scores
    bm25_raw_scores = bm25_db.get_scores(tokenized_query)
    bm25_top_indices = sorted(
        range(len(bm25_raw_scores)), key=lambda i: bm25_raw_scores[i], reverse=True
    )[:top_k]
    bm25_scores = [bm25_raw_scores[i] for i in bm25_top_indices]

    # Normalize BM25 scores
    bm25_scaled = MinMaxScaler().fit_transform([[s] for s in bm25_scores])
    bm25_normalized = [s[0] for s in bm25_scaled]

    # 3. Weighted score map
    score_map = defaultdict(float)
    meta_map = {}

    # Add FAISS scores
    for i, doc in enumerate(faiss_docs):
        key = doc.page_content
        score_map[key] += w1 * faiss_normalized[i]
        meta_map[key] = doc.metadata

    # Add BM25 scores
    for i, idx in enumerate(bm25_top_indices):
        doc = chunks[idx]
        key = doc.page_content
        score_map[key] += w2 * bm25_normalized[i]
        meta_map[key] = doc.metadata

    # 4. Rank combined scores
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for rank, (content, score) in enumerate(ranked, 1):
        results.append(
            {
                "rank": rank,
                "score": round(score, 4),
                "timestamp": meta_map[content].get("timestamp", "N/A"),
                "title": meta_map[content].get("title", "N/A"),
                "uploader": meta_map[content].get("uploader", "N/A"),
                "upload_date": meta_map[content].get("upload_date", "N/A"),
                "description": meta_map[content].get("description", ""),
                "text": content,
            }
        )

    return results


def prepare_context(retreived_chunks):
    if not retreived_chunks:
        return ""

    # Extract metadata from the first chunk (all chunks share the same video metadata)
    meta = retreived_chunks[0]
    title = meta.get("title", "Unknown Title")
    uploader = meta.get("uploader", "Unknown Uploader")
    upload_date = meta.get("upload_date", "Unknown Date")
    description = meta.get("description", "")

    context_parts = [
        f"Video Title: {title}",
        f"Uploader: {uploader}",
        f"Upload Date: {upload_date}",
        f"Description: {description}",
        "\n---\n",
    ]

    for chunk in retreived_chunks:
        content = chunk["text"]
        timestamp = chunk["timestamp"]
        context_parts.append(f"Content: {content}\nTimestamp: {timestamp}\n")

    return "\n\n".join(context_parts)


# query = "What is this section about?"
# retreived_chunks = hybrid_ranked_chunks(query, faiss_db, bm25_db, chunk_texts, chunks, top_k = 5)
