import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from pdf_processor import detect_sections, extract_text_images, save_pdf_pages
import fitz

PDF_DIR = "data/pdfs"
OUTPUT_DIR = "outputs"
CHUNK_SIZE = 50

# Initialize Model
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension() # FAISS dimension definition

def embed_sections(doc, sections_list, pdf_name):
    vector_metadata = []
    if not sections_list:
        # Fallback: treat each page as its own section
        sections_list = [{"header": f"Page_{i+1}", "page": i, "level": "section"} for i in range(len(doc))]

    # Collect all section headers for building header_terms context
    all_headers_by_page = {}
    for sec in sections_list:
        all_headers_by_page.setdefault(sec["page"], []).append(sec["header"])

    for i, sec in enumerate(sections_list):
        start_page = sec["page"]
        end_page = sections_list[i+1]["page"]-1 if i+1 < len(sections_list) else len(doc)-1
        text, images = extract_text_images(doc, start_page, end_page, pdf_name)
        sentences = sent_tokenize(text)
        if not sentences: continue

        # Build header_terms: this header, and any previous/parent header on the starting page
        header_terms = [sec["header"]]
        # You can add more context here if needed:
        if i > 0: header_terms.append(sections_list[i-1]["header"])
        # Add all section headers from the chunk range:
        for pg in range(start_page, end_page+1):
            header_terms.extend(all_headers_by_page.get(pg, []))
        header_terms = list(set(header_terms))

        for j in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = " ".join(sentences[j:j+CHUNK_SIZE])
            vector_text = f"{sec['header']} {chunk_text}"
            embedding = model.encode([vector_text])

            vector_metadata.append({
                "pdf_name": pdf_name,
                "header_terms": header_terms,
                "vector_id": f"{pdf_name}_{sec['header'].replace(' ','_')}_chunk{j//CHUNK_SIZE}",
                "vector_text": vector_text,
                "images": images,
                "page_range": [start_page+1, end_page+1],
                "level": sec["level"],
                "embedding": embedding.tolist()
            })          

    return vector_metadata

# Process All PDFs
all_vectors = []
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    print(f"\n📘 Processing: {pdf_file}")
    doc = fitz.open(pdf_path)
    save_pdf_pages(doc, pdf_file.replace(".pdf",""))
    sections = detect_sections(doc)
    vectors = embed_sections(doc, sections, pdf_file.replace(".pdf",""))
    if vectors:
        all_vectors.extend(vectors)
        print(f" → {len(vectors)} vectors created")
    else:
        print(f" → Skipped (0 vectors)")

print(f"\n✅ Total vectors: {len(all_vectors)}")

# Save Metadata
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(all_vectors, f, ensure_ascii=False, indent=2)
print(f"📄 Metadata saved to {metadata_path}")

# Optional: FAISS indexes (not always needed for pure RAG retrieval, but useful for batch search)
section_index = faiss.IndexFlatL2(embedding_dim)
subsection_index = faiss.IndexFlatL2(embedding_dim)
for meta in all_vectors:
    vec = np.array(meta["embedding"], dtype=np.float32)
    if vec.ndim == 1: vec = vec.reshape(1, -1)
    if meta["level"] == "section":
        section_index.add(vec)
    else:
        subsection_index.add(vec)
print("🔍 FAISS indexes ready.")
