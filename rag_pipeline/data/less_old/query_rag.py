import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

METADATA_PATH = "outputs/metadata.json"
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_vectors = json.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")

def robust_header_score(chunk, query):
    score = 0
    pdf_name = chunk.get("pdf_name", "").lower()
    if pdf_name and pdf_name in query.lower():
        score += 2
    header_terms = chunk.get("header_terms", [])
    if isinstance(header_terms, str):
        header_terms = header_terms.split(" | ")
    query_lower = query.lower()
    query_words = set(query_lower.split())
    header_keywords = set()
    for term in header_terms:
        term_lower = term.lower()
        term_words = set(term_lower.split())
        header_keywords.update(term_words)
        if (term_lower in query_lower) or (query_lower in term_lower):
            score += 2
        if term.isupper() and any(k in query_lower for k in ['safety', 'warning', 'caution', 'read prior', 'important']):
            score += 2
    score += len(query_words.intersection(header_keywords))
    return score

def hybrid_rag(query, all_vectors, model, top_n=10, alpha=0.7):
    query_vec = model.encode([query])
    chunk_embs = np.array([c["embedding"][0] if isinstance(c["embedding"][0], (list, np.ndarray)) else c["embedding"] for c in all_vectors])
    sims = np.dot(chunk_embs, query_vec[0])

    manual_scores = np.array([robust_header_score(c, query) for c in all_vectors])
    norm_sims = (sims - np.min(sims)) / (np.ptp(sims) + 1e-8)
    norm_manual = (manual_scores - np.min(manual_scores)) / (np.ptp(manual_scores) + 1e-8)
    hybrid_score = alpha * norm_sims + (1 - alpha) * norm_manual
    idxs = np.argsort(hybrid_score)[::-1][:top_n]
    return [all_vectors[i] for i in idxs]

if __name__ == "__main__":
    query = "What is the default IP address for the InsightPower Mini SNMP Card"
    results = hybrid_rag(query, all_vectors, model, top_n=5, alpha=0.7)
    for r in results:
        print(f"Vector ID: {r['vector_id']}")
        print(f"PDF Name: {r.get('pdf_name', '')}")
        print(f"Header Terms: {r.get('header_terms', '')}")
        print(f"Level: {r.get('level', '')}")
        print(f"Text Snippet: {r['vector_text'][:200]}...")
        print(f"Page Range: {r['page_range']}")
        print("------")
