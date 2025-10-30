import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm

# ================= CONFIG =================
parser = argparse.ArgumentParser()
parser.add_argument("--eval", type=str, default="qrels.json", help="Path to qrels.json")
parser.add_argument("--meta_dir", type=str, default="outputs", help="Directory containing metadata JSONs")
TOP_TEXT = 7
TOP_HEADER = 3
args = parser.parse_args()

TEXT_META_PATH = os.path.join(args.meta_dir, "metadata_text.json")
HEADER_META_PATH = os.path.join(args.meta_dir, "metadata_headers.json")
QRELS_PATH = args.eval
# ==========================================

# Load model
print("🔍 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load metadata
with open(TEXT_META_PATH, "r", encoding="utf-8") as f:
    text_meta = json.load(f)
with open(HEADER_META_PATH, "r", encoding="utf-8") as f:
    header_meta = json.load(f)

for m in text_meta:
    m["embedding"] = np.array(m["embedding"], dtype=np.float32).flatten()
for h in header_meta:
    h["embedding"] = np.array(h["embedding"], dtype=np.float32).flatten()

# --- Build FAISS indexes ---
def build_index(meta_list):
    dim = len(meta_list[0]["embedding"])
    index = faiss.IndexFlatIP(dim)
    embeddings = np.vstack([m["embedding"].reshape(1, -1) for m in meta_list])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

header_index = build_index(header_meta)
text_index = build_index(text_meta)

# --- Query ---
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

# --- Load Qrels ---
with open(QRELS_PATH, "r", encoding="utf-8") as f:
    qrels = json.load(f)

# --- Evaluate ---
results = []
recalls = []

for q in tqdm(qrels, desc="Evaluating queries"):
    query = q["query"]
    ground_truths = q["ground_truth"]

    top_headers, top_texts = retrieve_top_k(query, model, text_meta, header_meta, TOP_TEXT, TOP_HEADER)

    retrieved_pdfs = set([x["pdf_name"] for x in top_headers + top_texts])
    gt_pdfs = set([g["pdf_name"] for g in ground_truths])

    hit = bool(retrieved_pdfs.intersection(gt_pdfs))
    recall = 1.0 if hit else 0.0
    recalls.append(recall)

    results.append({
        "query": query,
        "retrieved_pdfs": list(retrieved_pdfs),
        "ground_truth_pdfs": list(gt_pdfs),
        "hit": hit,
        "top_headers": [{"pdf_name": h["pdf_name"], "pages": h["page_range"]} for h in top_headers],
        "top_texts": [{"pdf_name": t["pdf_name"], "pages": t["page_range"]} for t in top_texts],
    })

avg_recall = np.mean(recalls)
print(f"\n📊 Average Recall@{TOP_TEXT + TOP_HEADER}: {avg_recall:.3f}")

# --- Save detailed results ---
os.makedirs(args.meta_dir, exist_ok=True)
with open(os.path.join(args.meta_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Detailed evaluation results saved to {args.meta_dir}/evaluation_results.json")
# --- Print evaluation results ---
print("\n=== Evaluation Results ===")
for r in results:
    print(f"\nQuery: {r['query']}")
    print("Top Headers:")
    for h in r["top_headers"]:
        print(f"  PDF: {h['pdf_name']} | Pages: {h['pages']}")
    print("Top Text Chunks:")
    for t in r["top_texts"]:
        print(f"  PDF: {t['pdf_name']} | Pages: {t['pages']}")