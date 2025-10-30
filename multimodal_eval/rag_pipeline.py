# ===========================================
# Multi-PDF RAG Pipeline with Flexible Section Detection + Page Extraction
# ===========================================

import os
import re
import fitz
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import json
from PIL import Image
import io

nltk.download("punkt")
nltk.download("punkt_tab")

# ---------------------------
# 1. Config
# ---------------------------
PDF_DIR = "data/pdfs"       # Folder containing your PDFs
OUTPUT_DIR = "outputs"
CHUNK_SIZE = 300
TOP_SECTIONS = 2
TOP_SUBSECTIONS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pages"), exist_ok=True)

# ---------------------------
# 2. Initialize Model
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384

# ---------------------------
# 3. Image Filtering Functions
# ---------------------------
def is_valid_image(img_bytes, min_width=50, min_height=50):
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    return w >= min_width and h >= min_height

def is_not_blackbox(img_bytes, black_thresh=10):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    arr = np.array(img)
    avg_pixel = arr.mean()
    return avg_pixel > black_thresh

def has_normal_aspect(img_bytes, min_ratio=0.5, max_ratio=2.0):
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    ratio = w / h
    return min_ratio <= ratio <= max_ratio

# ---------------------------
# 4. Flexible Section Detection
# ---------------------------
def detect_sections(doc):
    sections = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = span["size"]

                    # Numbered header pattern (dot or dash)
                    pattern_numbered = re.match(r"^\d+[-\.]\d+([-\.\d+]*)?\s", text)

                    # Flexible detection: numbered or large-font
                    if size >= 12 and (pattern_numbered or len(text.split()) > 3):
                        if pattern_numbered:
                            level = "subsection" if text.count('.') + text.count('-') > 1 else "section"
                        else:
                            level = "section"
                        sections.append({
                            "header": text,
                            "page": page_num,
                            "level": level
                        })
    return sections

# ---------------------------
# 5. Extract Text + Images
# ---------------------------
def extract_text_images(doc, start_page, end_page, pdf_name):
    text_content = ""
    pdf_img_dir = os.path.join(OUTPUT_DIR, "images", pdf_name)
    os.makedirs(pdf_img_dir, exist_ok=True)
    images = []

    for p in range(start_page, end_page + 1):
        page = doc[p]
        text_content += page.get_text("text") + "\n"

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]

            if not (is_valid_image(img_bytes) and is_not_blackbox(img_bytes) and has_normal_aspect(img_bytes)):
                continue

            img_path = os.path.join(pdf_img_dir, f"page{p+1}_img{img_index}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            images.append(img_path)

    return text_content, images

# ---------------------------
# 6. Save Individual PDF Pages
# ---------------------------
def save_pdf_pages(doc, pdf_name):
    pdf_dir = os.path.join(OUTPUT_DIR, "pages", pdf_name)
    os.makedirs(pdf_dir, exist_ok=True)

    for i, page in enumerate(doc):
        pdf_path = os.path.join(pdf_dir, f"page_{i+1}.pdf")
        new_doc = fitz.open()        # create empty PDF
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        new_doc.save(pdf_path)
        new_doc.close()

# ---------------------------
# 7. Embed Sections/Subsections
# ---------------------------
def embed_sections(doc, sections_list, pdf_name):
    vector_metadata = []
    if not sections_list:
        # fallback: treat each page as a section
        sections_list = [{"header": f"Page_{i+1}", "page": i, "level": "section"} for i in range(len(doc))]

    for i, sec in enumerate(sections_list):
        start_page = sec["page"]
        end_page = sections_list[i+1]["page"]-1 if i+1 < len(sections_list) else len(doc)-1

        text, images = extract_text_images(doc, start_page, end_page, pdf_name)
        sentences = sent_tokenize(text)
        if not sentences:
            continue

        for j in range(0, len(sentences), CHUNK_SIZE):
            chunk_text = " ".join(sentences[j:j+CHUNK_SIZE])
            vector_text = f"{sec['header']} {chunk_text}"
            embedding = model.encode([vector_text])

            vector_metadata.append({
                "vector_id": f"{pdf_name}_{sec['header'].replace(' ','_')}_chunk{j//CHUNK_SIZE}",
                "vector_text": vector_text,
                "images": images,
                "page_range": (start_page+1, end_page+1),
                "level": sec["level"],
                "embedding": embedding.tolist(),
                "pdf_name": pdf_name
            })
    return vector_metadata

# ---------------------------
# 8. Process All PDFs
# ---------------------------
all_vectors = []
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    print(f"\n📘 Processing: {pdf_file}")
    doc = fitz.open(pdf_path)

    # Save individual pages
    save_pdf_pages(doc, pdf_file.replace(".pdf",""))

    sections = detect_sections(doc)
    vectors = embed_sections(doc, sections, pdf_file.replace(".pdf",""))
    if vectors:
        all_vectors.extend(vectors)
        print(f"  → {len(vectors)} vectors created")
    else:
        print(f"  → Skipped (0 vectors)")

print(f"\n✅ Total vectors across all PDFs: {len(all_vectors)}")

# ---------------------------
# 9. Build FAISS Indexes
# ---------------------------
section_index = faiss.IndexFlatL2(embedding_dim)
subsection_index = faiss.IndexFlatL2(embedding_dim)

for meta in all_vectors:
    vec = np.array(meta["embedding"], dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if meta["level"] == "section":
        section_index.add(vec)
    else:
        subsection_index.add(vec)

print("🔍 FAISS indexes ready.")

# ---------------------------
# 10. Save Metadata
# ---------------------------
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(all_vectors, f, ensure_ascii=False, indent=2)
print(f"📄 Metadata saved to {metadata_path}")

# ---------------------------
# 11. Hierarchical Query
# ---------------------------
def hierarchical_query(query_text, top_sections=TOP_SECTIONS, top_subsections=TOP_SUBSECTIONS):
    query_vec = model.encode([query_text])
    query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)

    # Step 1: Search sections
    distances, indices = section_index.search(query_vec, top_sections)
    retrieved_sections = [all_vectors[i] for i in indices[0]]

    sec_start = retrieved_sections[0]["page_range"][0]
    sec_end = retrieved_sections[-1]["page_range"][1]

    # Step 2: Candidate subsections
    candidate_subs = [
        s for s in all_vectors
        if s["level"]=="subsection" and s["page_range"][0]>=sec_start and s["page_range"][1]<=sec_end
    ]
    if not candidate_subs:
        candidate_subs = [s for s in all_vectors if s["level"]=="subsection"]

    # Step 3: Build temp FAISS for candidate subsections
    temp_index = faiss.IndexFlatL2(embedding_dim)
    for s in candidate_subs:
        vec = np.array(s["embedding"], dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        temp_index.add(vec)

    distances, indices = temp_index.search(query_vec, top_subsections)
    return [candidate_subs[idx] for idx in indices[0]]

# ---------------------------
# 12. Example Query
# ---------------------------
query = "How do I clean the cooling fan in the PMA-SynRM?"
results = hierarchical_query(query)

print("\n🔎 Query Results:")
for r in results:
    print("Vector ID:", r["vector_id"])
    print("Text Snippet:", r["vector_text"][:250], "...")
    print("Images:", r["images"])
    print("Page Range:", r["page_range"])
    print("Level:", r["level"])
    print("PDF:", r["pdf_name"])
    print("------")
