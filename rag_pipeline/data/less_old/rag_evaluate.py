import json
from query_rag import hybrid_rag, all_vectors, model  # Import appropriately for your file structure

# ---------------------------
# Config
# ---------------------------
QRELS_PATH = "qrels.json"
TOP_RESULTS = 3  # Number of retrievals to consider per query
ALPHA = 0.7      # 0.7 = semantic > header; 0.5 = equal; lower = more boost by metadata

# ---------------------------
# Load Qrels
# ---------------------------
with open(QRELS_PATH, "r", encoding="utf-8") as f:
    qrels = json.load(f)

# ---------------------------
# Helper
# ---------------------------
def page_overlap(gt_range, retrieved_range):
    """Return True if the retrieved page range overlaps with ground truth."""
    return not (retrieved_range[1] < gt_range[0] or retrieved_range[0] > gt_range[1])

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_rag(qrels, all_vectors, model, top_results=TOP_RESULTS, alpha=ALPHA):
    correct = 0
    failed_queries = []
    for entry in qrels:
        query = entry["query"]
        gt_list = entry["ground_truth"]

        results = hybrid_rag(query, all_vectors, model, top_n=top_results, alpha=alpha)

        hit = False
        for gt in gt_list:
            for r in results:
                if r["pdf_name"] == gt["pdf_name"] and page_overlap(gt["page_range"], r["page_range"]):
                    hit = True
                    break
            if hit:
                break

        if hit:
            correct += 1
        else:
            failed_queries.append({
                "query": query,
                "ground_truth": gt_list,
                "retrieved": [{
                    "vector_id": r["vector_id"],
                    "pdf_name": r["pdf_name"],
                    "page_range": r["page_range"],
                    "score": getattr(r, 'score', None)  # robust_header_score if present
                } for r in results]
            })

    accuracy = correct / len(qrels)
    return accuracy, failed_queries, results

# ---------------------------
# Run Evaluation
# ---------------------------
accuracy, failed_queries, results = evaluate_rag(qrels, all_vectors, model)
print(f"\nHybrid RAG retrieval accuracy: {accuracy:.2%}")

if failed_queries:
    print("\nFailed Queries Detail:")
    for fq in failed_queries:
        print(f"\nQuery: {fq['query']}")
        print(f"Ground Truth: {fq['ground_truth']}")
        print(f"Retrieved Vectors: {fq['retrieved']}")
