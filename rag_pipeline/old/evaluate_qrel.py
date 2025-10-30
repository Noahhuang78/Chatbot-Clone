# rag_eval.py
import json
from rag_pipeline.old.query_rag import hierarchical_query  # replace with your actual query module

# ---------------------------
# Config
# ---------------------------
QRELS_PATH = "qrels.json"
TOP_SECTIONS = 2
TOP_SUBSECTIONS = 3

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
def evaluate_rag(qrels, top_sections=TOP_SECTIONS, top_subsections=TOP_SUBSECTIONS):
    correct = 0
    failed_queries = []

    for entry in qrels:
        query = entry["query"]
        gt_list = entry["ground_truth"]

        results = hierarchical_query(query, top_sections, top_subsections)

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
                    "page_range": r["page_range"]
                } for r in results]
            })

    accuracy = correct / len(qrels)
    return accuracy, failed_queries, results

# ---------------------------
# Run Evaluation
# ---------------------------
accuracy, failed_queries, results = evaluate_rag(qrels)
print(f"\nHierarchical RAG retrieval accuracy: {accuracy:.2%}")

if failed_queries:
    print("\nFailed Queries Detail:")
    for fq in failed_queries:
        print(f"\nQuery: {fq['query']}")
        print(f"Ground Truth: {fq['ground_truth']}")
        print(f"Retrieved Vectors: {fq['retrieved']}")
