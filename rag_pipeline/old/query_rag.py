import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

# Load metadata
METADATA_PATH = "outputs/metadata.json"
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_vectors = json.load(f)

# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")

def robust_header_score(vector, query):
    """Enhance header score using pdf_name, section titles, and vector_id chunks."""
    score = 0
    # pdf_name context matching
    pdf_name = vector.get("pdf_name", "").lower()
    if pdf_name and pdf_name in query.lower():
        score += 2
    # section or chunk header matching
    header_terms = []
    if "header_terms" in vector:  # If available from metadata
        header_terms = vector["header_terms"]
    else:
        # Attempt to extract from vector_id (fallback)
        header_terms = vector["vector_id"].replace(".pdf", "").split("_")[1:]
    for term in header_terms:
        if term.lower() in query.lower():
            score += 1

    return score

def hierarchical_query(query, top_sections=3, top_subsections=5):
    query_vec = model.encode([query])

    # Step 1: Filter by likely PDF/document context
    relevant_vectors = [
        v for v in all_vectors
        if query.lower().split()[0] in v.get("pdf_name", "").lower()  # crude: uses first query term, refine if needed
        or any(term in v.get("pdf_name", "").lower() for term in query.lower().split())
        or ("mini snmp" in query.lower() and "snmp" in v.get("pdf_name", "").lower())
    ]
    # Fallback: if filtering is too strict, use all_vectors
    if not relevant_vectors:
        relevant_vectors = all_vectors

    # Step 2: Top-level section search
    section_vectors = [v for v in relevant_vectors if v['level'] == 'section']
    if not section_vectors:
        section_vectors = relevant_vectors

    section_embeddings = np.array([np.array(v['embedding'][0], dtype=np.float32) for v in section_vectors])
    index = faiss.IndexFlatIP(section_embeddings.shape[1])
    index.add(section_embeddings)
    D, I = index.search(np.array(query_vec, dtype=np.float32), min(top_sections, len(section_vectors)))
    top_section_vectors = [section_vectors[i] for i in I[0]]

    # Step 3: Boost by robust header score
    for v in top_section_vectors:
        v['score'] = 1 + robust_header_score(v, query)

    # Step 4: Subsection retrieval inside top sections
    results = []
    for sec in top_section_vectors:
        sec_start, sec_end = sec['page_range']
        subsections = [
            v for v in relevant_vectors
            if v['level'] != 'section'
            and v['page_range'][0] >= sec_start
            and v['page_range'][1] <= sec_end
            and v.get('pdf_name', '') == sec.get('pdf_name', '')  # Same PDF for safety
        ]
        if not subsections:
            results.append(sec)
            continue

        sub_emb = np.array([np.array(v['embedding'][0], dtype=np.float32) for v in subsections])
        idx = faiss.IndexFlatL2(sub_emb.shape[1])
        idx.add(sub_emb)
        D_sub, I_sub = idx.search(np.array(query_vec, dtype=np.float32), min(top_subsections, len(subsections)))
        for i in I_sub[0]:
            sub = subsections[i]
            sub['score'] = 1 + robust_header_score(sub, query)
            results.append(sub)

    # Step 5: Final sorting by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return results

# Optional: quick test
if __name__ == "__main__":
    query = "What is the default IP address for the InsightPower Mini SNMP Card?"
    results = hierarchical_query(query)
    for r in results:
        print(f"Vector ID: {r['vector_id']}")
        print(f"PDF Name: {r.get('pdf_name', '')}")
        print(f"Text Snippet: {r['vector_text'][:200]}...")
        print(f"Page Range: {r['page_range']}")
        print(f"Score: {r.get('score', 0)}")
        print("------")
