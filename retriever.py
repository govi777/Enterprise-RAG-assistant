import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Model singletons (lazy-loaded once per process) ───────────────────────────
_bi_encoder = None
_cross_encoder = None
_index = None
_chunks = None


def _get_bi_encoder():
    global _bi_encoder
    if _bi_encoder is None:
        _bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _bi_encoder


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return _cross_encoder


def _load_index():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index("vector.index")
        with open("chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)
    return _index, _chunks


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query, top_k=5, rerank_top_k=3, score_threshold=0.15):
    """
    Two-stage retrieval:
      1. Bi-encoder FAISS search (fast, recall-oriented)
      2. Cross-encoder rerank (slow, precision-oriented)

    Returns list of dicts with keys:
        chunk, source, page, bi_score, rerank_score
    """
    index, chunks = _load_index()
    bi_encoder = _get_bi_encoder()
    cross_encoder = _get_cross_encoder()

    # Stage 1 – dense retrieval
    query_emb = bi_encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, top_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < score_threshold:
            continue                          # drop low-relevance chunks
        entry = chunks[idx]
        # Support both old format (plain str) and new format (dict with metadata)
        if isinstance(entry, dict):
            candidates.append({
                "chunk": entry["chunk"],
                "source": entry.get("source", "company_docs.pdf"),
                "page": entry.get("page", "—"),
                "bi_score": float(score),
                "rerank_score": 0.0,
            })
        else:
            candidates.append({
                "chunk": entry,
                "source": "company_docs.pdf",
                "page": "—",
                "bi_score": float(score),
                "rerank_score": 0.0,
            })

    if not candidates:
        return []

    # Stage 2 – cross-encoder rerank
    pairs = [[query, c["chunk"]] for c in candidates]
    rerank_scores = cross_encoder.predict(pairs)
    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:rerank_top_k]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    question = input("Enter your question: ")
    results = retrieve(question)
    print(f"\nTop {len(results)} results:\n")
    for r in results:
        print(f"[{r['source']} p.{r['page']}]  rerank={r['rerank_score']:.3f}  bi={r['bi_score']:.3f}")
        print(r["chunk"])
        print("-" * 50)
