import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from pdf_processor import detect_sections, extract_text_images, save_pdf_pages
import fitz

PDF_DIR = "data/pdfs"
OUTPUT_DIR = "outputs"
CHUNK_SIZE = 50

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()


def embed_sections(doc, sections_list, pdf_name):
    vector_metadata = []
    header_vectors = []

    if not sections_list:
        sections_list = [{"header": f"Page_{i+1}", "page": i, "level": "section"} for i in range(len(doc))]

    all_headers_by_page = {}
    for sec in sections_list:
        all_headers_by_page.setdefault(sec["page"], []).append(sec["header"])

    for i, sec in enumerate(sections_list):
        start_page = sec["page"]
        end_page = sections_list[i+1]["page"] - 1 if i+1 < len(sections_list) else len(doc)-1
        section_text, images = extract_text_images(doc, start_page, end_page, pdf_name)
        sentences = sent_tokenize(section_text)
        if not sentences:
            continue

        # --- Header embedding (flattened) ---
        header_text = sec["header"]
        header_embedding = model.encode([header_text]).flatten()
        header_vectors.append({
            "pdf_name": pdf_name,
            "header": header_text,
            "vector_id": f"{pdf_name}_header_{i}",
            "vector_text": section_text,      # full section text
            "embedding": header_embedding.tolist(),
            "level": sec["level"],
            "page_range": [start_page+1, end_page+1],
            "images": images
        })

        # --- Text chunk embeddings ---
        header_terms = [sec["header"]]
        if i > 0:
            header_terms.append(sections_list[i-1]["header"])
        for pg in range(start_page, end_page+1):
            header_terms.extend(all_headers_by_page.get(pg, []))
        header_terms = list(set(header_terms))

        for j in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = " ".join(sentences[j:j+CHUNK_SIZE])
            vector_text = f"{sec['header']} {chunk_text}"
            embedding = model.encode([vector_text]).flatten()
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

    return vector_metadata, header_vectors


# --- Process PDFs ---
all_vectors = []
all_header_vectors = []

pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    print(f"\n📘 Processing: {pdf_file}")
    doc = fitz.open(pdf_path)
    save_pdf_pages(doc, pdf_file.replace(".pdf",""))
    sections = detect_sections(doc)
    vectors, header_vecs = embed_sections(doc, sections, pdf_file.replace(".pdf",""))
    if vectors:
        all_vectors.extend(vectors)
        all_header_vectors.extend(header_vecs)
        print(f" → {len(vectors)} text vectors | {len(header_vecs)} header vectors created")
    else:
        print(f" → Skipped (0 vectors)")

# --- Save metadata ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "metadata_text.json"), "w", encoding="utf-8") as f:
    json.dump(all_vectors, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUTPUT_DIR, "metadata_headers.json"), "w", encoding="utf-8") as f:
    json.dump(all_header_vectors, f, ensure_ascii=False, indent=2)
print("📄 Metadata saved.")
