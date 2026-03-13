# 📚 Enterprise Knowledge Assistant (RAG)

A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions against company documents. Built with FAISS, Sentence Transformers, a Cross-Encoder reranker, OpenAI GPT-4o-mini, and a Streamlit UI.

---

## 🖥️ Demo

> Ask questions like:
> - *"What is the work from home policy?"*
> - *"What is the notice period for senior engineers?"*
> - *"What are the password requirements?"*
> - *"What is the meal reimbursement limit?"*

The assistant retrieves the most relevant chunks from your documents and generates a grounded answer with source citations.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Bi-Encoder (all-MiniLM-L6-v2)
    │  FAISS vector search → top 5 candidates
    ▼
Cross-Encoder (ms-marco-MiniLM-L-6-v2)
    │  Rerank → top 3 chunks
    ▼
GPT-4o-mini
    │  Answer grounded in retrieved context
    ▼
Streamlit UI (streamed response)
```

---

## ✨ Features

- **Smart chunking** — sentence-aware splitting, no cut-off words
- **Two-stage retrieval** — bi-encoder for recall, cross-encoder for precision
- **Source citations** — every answer references the document and page number
- **Streaming responses** — answers stream token by token
- **Multi-format document support** — PDF, DOCX, TXT
- **On-the-fly ingestion** — upload new documents directly from the UI
- **Conversation memory** — follow-up questions are supported
- **👍 / 👎 feedback** — rate answers, logged to `query_log.jsonl`
- **Session history** — previous Q&A pairs visible in the same session
- **Startup validation** — clear error if API key is missing

---

## 📁 Project Structure

```
RAG/
├── app.py              # Streamlit frontend
├── ingest.py           # Document loading, chunking, embedding, FAISS indexing
├── retriever.py        # Bi-encoder retrieval + cross-encoder reranking
├── generator.py        # Prompt builder + OpenAI streaming response
├── data/
│   └── company_docs.pdf
├── vector.index        # FAISS index (auto-generated)
├── chunks.pkl          # Chunked text with metadata (auto-generated)
├── query_log.jsonl     # Query + feedback log (auto-generated)
├── requirements.txt
└── .env
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/RAG.git
cd RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-key-here
```

### 4. Add your document

Place your PDF (or DOCX / TXT) inside the `data/` folder.

### 5. Build the vector index

```bash
python ingest.py
```

### 6. Run the app

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Bi-encoder + cross-encoder models |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `openai` | GPT-4o-mini API |
| `streamlit` | Web UI |
| `python-dotenv` | Environment variable loading |

---

## 🔄 Re-ingesting Documents

Every time you change or add a document, re-run ingestion to rebuild the index:

```bash
python ingest.py
```

Or upload directly from the sidebar in the app — it ingests automatically.

---

## 📝 Query Log

All queries, answers, sources, and feedback are saved to `query_log.jsonl`:

```json
{
  "ts": "2025-03-13T10:22:01",
  "question": "What is the WFH policy?",
  "answer": "Employees may work remotely up to 3 days per week...",
  "sources": [{"source": "company_docs.pdf", "page": 2, "rerank": 4.21}],
  "feedback": "👍"
}
```

---

## 🚀 Future Improvements

- [ ] Support for multiple documents simultaneously
- [ ] Persistent vector store (ChromaDB / Pinecone)
- [ ] Authentication for enterprise use
- [ ] Docker deployment

---

## 🛡️ .gitignore Reminder

Make sure your `.gitignore` includes:

```
.env
__pycache__/
vector.index
chunks.pkl
query_log.jsonl
data/
```

---

## 👤 Author

**Govind**  
Computer Science Graduate  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)
