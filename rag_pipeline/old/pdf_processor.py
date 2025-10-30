import os
import re
import fitz
import nltk
from nltk.tokenize import sent_tokenize
from PIL import Image
import io
import numpy as np

nltk.download("punkt")
nltk.download("punkt_tab")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pages"), exist_ok=True)

# ---------------------------
# Image Filtering
# ---------------------------
def is_valid_image(img_bytes, min_width=50, min_height=50):
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    return w >= min_width and h >= min_height

def is_not_blackbox(img_bytes, black_thresh=10):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    return np.array(img).mean() > black_thresh

def has_normal_aspect(img_bytes, min_ratio=0.5, max_ratio=2.0):
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size
    ratio = w / h
    return min_ratio <= ratio <= max_ratio

# ---------------------------
# Section Detection
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

                    pattern_numbered = re.match(r"^\d+[-\.]\d+([-\.\d+]*)?\s", text)
                    if size >= 12 and (pattern_numbered or len(text.split()) > 3):
                        if pattern_numbered:
                            level = "subsection" if text.count('.') + text.count('-') > 1 else "section"
                        else:
                            level = "section"
                        sections.append({"header": text, "page": page_num, "level": level})
    return sections

# ---------------------------
# Extract Text + Images
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
# Save PDF Pages
# ---------------------------
def save_pdf_pages(doc, pdf_name):
    pdf_dir = os.path.join(OUTPUT_DIR, "pages", pdf_name)
    os.makedirs(pdf_dir, exist_ok=True)
    for i, page in enumerate(doc):
        pdf_path = os.path.join(pdf_dir, f"page_{i+1}.pdf")
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        new_doc.save(pdf_path)
        new_doc.close()
