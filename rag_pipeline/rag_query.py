import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Config ---
META_DIR = "outputs"
TOP_TEXT = 7
TOP_HEADER = 3

TEXT_META_PATH = os.path.join(META_DIR, "metadata_text.json")
HEADER_META_PATH = os.path.join(META_DIR, "metadata_headers.json")

# --- Load model ---
print("🔍 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load metadata ---
with open(TEXT_META_PATH, "r", encoding="utf-8") as f:
    text_meta = json.load(f)
with open(HEADER_META_PATH, "r", encoding="utf-8") as f:
    header_meta = json.load(f)

# Convert embeddings to np.array and flatten
for m in text_meta:
    m["embedding"] = np.array(m["embedding"], dtype=np.float32).flatten()
for h in header_meta:
    h["embedding"] = np.array(h["embedding"], dtype=np.float32).flatten()

# --- Build FAISS Indexes ---
def build_index(meta_list):
    dim = len(meta_list[0]["embedding"])
    index = faiss.IndexFlatIP(dim)
    embeddings = np.vstack([m["embedding"].reshape(1, -1) for m in meta_list])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

header_index = build_index(header_meta)
text_index = build_index(text_meta)

# --- Query function ---
def retrieve_top_k(query, model, text_meta, header_meta, top_text=TOP_TEXT, top_header=TOP_HEADER):
    query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    faiss.normalize_L2(query_emb)

    # Headers
    D, I = header_index.search(query_emb, top_header)
    top_headers = []
    for idx in I[0]:
        h = header_meta[idx]
        top_headers.append({
            "pdf_name": h["pdf_name"],
            "text": h["vector_text"],
            "page_range": h["page_range"],
            "images": h["images"]
        })

    # Text chunks
    D, I = text_index.search(query_emb, top_text)
    top_texts = []
    for idx in I[0]:
        t = text_meta[idx]
        top_texts.append({
            "pdf_name": t["pdf_name"],
            "text": t["vector_text"],
            "page_range": t["page_range"],
            "images": t["images"]
        })

    return top_headers, top_texts


# --- Example usage ---
if __name__ == "__main__":
    query = "What is the default IP address for the InsightPower Mini SNMP Card?"
    top_headers, top_texts = retrieve_top_k(query, model, text_meta, header_meta)

    print("\n=== Top 3 Headers ===")
    for h in top_headers:
        print(f"PDF: {h['pdf_name']} | Pages: {h['page_range']}")
        print(f"Text (first 300 chars): {h['text'][:300]}...")
        print(f"Images: {h['images']}\n")

    print("\n=== Top 7 Text Chunks ===")
    for t in top_texts:
        print(f"PDF: {t['pdf_name']} | Pages: {t['page_range']}")
        print(f"Text (first 300 chars): {t['text'][:300]}...")
        print(f"Images: {t['images']}\n")
