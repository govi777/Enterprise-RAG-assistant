import os
import faiss
import numpy as np
import pickle
import re
from sentence_transformers import SentenceTransformer


# ── Document loaders ──────────────────────────────────────────────────────────

def load_pdf(file_path):
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "page": i + 1, "source": os.path.basename(file_path)})
    return pages


def load_docx(file_path):
    import docx
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [{"text": text, "page": 1, "source": os.path.basename(file_path)}]


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [{"text": text, "page": 1, "source": os.path.basename(file_path)}]


def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt}
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")
    return loaders[ext](file_path)


# ── Smart sentence-aware chunking ─────────────────────────────────────────────

def smart_chunk(pages, chunk_size=500, overlap=100):
    """
    Splits text respecting sentence boundaries so chunks never
    start or end mid-sentence.
    """
    chunks = []
    for page_data in pages:
        text = page_data["text"].strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append({
                        "chunk": current_chunk.strip(),
                        "source": page_data["source"],
                        "page": page_data["page"],
                    })
                # Carry overlap from end of previous chunk
                words = current_chunk.split()
                overlap_text = " ".join(words[-(overlap // 6):]) if words else ""
                current_chunk = (overlap_text + " " + sentence).strip()
        if current_chunk:
            chunks.append({
                "chunk": current_chunk.strip(),
                "source": page_data["source"],
                "page": page_data["page"],
            })
    return chunks


# ── Embeddings & FAISS ────────────────────────────────────────────────────────

def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32
    )
    embeddings = np.array(embeddings)
    faiss.normalize_L2(embeddings)
    return embeddings


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    # IndexFlatIP = cosine similarity on L2-normalised vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def save_index(index, chunks, path="."):
    faiss.write_index(index, os.path.join(path, "vector.index"))
    with open(os.path.join(path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks.")


# ── Public entry point (called by app.py for on-the-fly ingestion) ────────────

def ingest_file(file_path, save_path="."):
    print(f"Loading {file_path} ...")
    pages = load_document(file_path)

    print("Chunking (sentence-aware) ...")
    chunks = smart_chunk(pages)
    print(f"  → {len(chunks)} chunks")

    print("Embedding ...")
    texts = [c["chunk"] for c in chunks]
    embeddings = create_embeddings(texts)

    print("Indexing ...")
    index = create_faiss_index(embeddings)

    print("Saving ...")
    save_index(index, chunks, path=save_path)
    print("Done!")
    return len(chunks)


if __name__ == "__main__":
    ingest_file("data/company_docs.pdf")
