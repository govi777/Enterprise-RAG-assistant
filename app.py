import os
import json
import tempfile
import datetime
import streamlit as st
from generator import stream_answer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Startup: validate API key early ──────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key.strip() == "your_actual_api_key_here":
    st.error(
        "⚠️ **OPENAI_API_KEY is missing or not set.**\n\n"
        "Add your key to the `.env` file:\n```\nOPENAI_API_KEY=sk-...\n```"
    )
    st.stop()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0f1117; color: #e8eaf0; }

[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #7eb8f7;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.rag-header {
    display: flex; align-items: center; gap: 14px;
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #2a2f3e;
    margin-bottom: 28px;
}
.rag-header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem; font-weight: 600; color: #ffffff;
}
.rag-header-sub { font-size: 0.82rem; color: #6b7280; margin-top: 2px; }

.pill-label {
    font-size: 0.72rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 8px; font-family: 'IBM Plex Mono', monospace;
}

.stTextInput > div > div > input {
    background-color: #1a1f2e !important;
    border: 1px solid #2a2f3e !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important; padding: 14px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7eb8f7 !important;
    box-shadow: 0 0 0 2px rgba(126,184,247,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #3d4355 !important; }

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    padding: 12px 28px !important; width: 100%;
    transition: opacity 0.15s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

.answer-card {
    background: linear-gradient(135deg, #1a2235, #1e2540);
    border: 1px solid #2e3a55; border-left: 3px solid #7eb8f7;
    border-radius: 10px; padding: 22px 24px; margin: 12px 0;
    font-size: 0.97rem; line-height: 1.7; color: #d4dff5;
}
.answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; color: #7eb8f7;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 6px;
}

.chunk-card {
    background-color: #141820; border: 1px solid #252b3b;
    border-radius: 8px; padding: 16px 18px; margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; line-height: 1.65; color: #8892a4;
}
.chunk-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
.chunk-num { font-size: 0.68rem; color: #7eb8f7; text-transform: uppercase; letter-spacing: 0.1em; }
.score-badge {
    background-color: #1e2d45; color: #60a5fa;
    font-size: 0.68rem; padding: 2px 8px; border-radius: 4px;
}
.source-tag {
    display: inline-block; background: #1a2a1a; color: #4ade80;
    font-size: 0.68rem; padding: 2px 8px; border-radius: 4px;
    margin-left: 6px; font-family: 'IBM Plex Mono', monospace;
}

.stat-box {
    background-color: #141820; border: 1px solid #1e2435;
    border-radius: 8px; padding: 14px 16px; text-align: center; margin-bottom: 8px;
}
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; color: #7eb8f7; }
.stat-label { font-size: 0.72rem; color: #4b5568; text-transform: uppercase; letter-spacing: 0.1em; }

.history-item {
    background-color: #141820; border: 1px solid #1e2435;
    border-radius: 7px; padding: 10px 13px; margin-bottom: 6px;
    font-size: 0.82rem; color: #6b7a96;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("current_q", ""),
    ("active_doc", "company_docs.pdf"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Logging helper ────────────────────────────────────────────────────────────
LOG_FILE = "query_log.jsonl"

def log_query(question, answer, sources, feedback=None):
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "sources": [{"source": s["source"], "page": s["page"], "rerank": s.get("rerank_score")} for s in sources],
        "feedback": feedback,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Document")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Ingesting document…"):
            try:
                from ingest import ingest_file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded.name)[1]
                ) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                n = ingest_file(tmp_path)
                os.unlink(tmp_path)
                st.session_state.active_doc = uploaded.name
                # Invalidate cached index so retriever reloads
                import retriever as _r
                _r._index = None
                _r._chunks = None
                st.success(f"✅ Ingested {n} chunks from **{uploaded.name}**")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown(
        f"<div style='background:#1a1f2e;border:1px solid #2a2f3e;border-radius:8px;"
        f"padding:12px 14px;font-size:0.82rem;color:#6b7280;"
        f"font-family:\"IBM Plex Mono\",monospace;'>"
        f"📄 {st.session_state.active_doc}<br>"
        f"<span style='color:#3d4355;font-size:0.72rem;'>FAISS · MiniLM-L6 · Cross-Encoder</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='stat-box'><div class='stat-value'>{len(st.session_state.history)}</div>"
            "<div class='stat-label'>Queries</div></div>", unsafe_allow_html=True
        )
    with c2:
        thumbs = sum(1 for h in st.session_state.history if h.get("feedback") == "👍")
        st.markdown(
            f"<div class='stat-box'><div class='stat-value'>{thumbs}</div>"
            "<div class='stat-label'>👍 Helpful</div></div>", unsafe_allow_html=True
        )

    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🕘 Recent")
        for item in reversed(st.session_state.history[-6:]):
            q_short = item["question"][:42] + ("…" if len(item["question"]) > 42 else "")
            st.markdown(f"<div class='history-item'>❯ {q_short}</div>", unsafe_allow_html=True)
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='rag-header'>"
    "<div style='font-size:2.2rem'>📚</div>"
    "<div><div class='rag-header-title'>Enterprise Knowledge Assistant</div>"
    "<div class='rag-header-sub'>Retrieval-Augmented Generation · Smart Chunking · Reranking</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "What is the work from home policy?",
    "What is the notice period for senior engineers?",
    "What are the password requirements?",
    "What is the meal reimbursement limit?",
]

st.markdown("<div class='pill-label'>💡 Suggested questions</div>", unsafe_allow_html=True)
cols = st.columns(len(SUGGESTIONS))
for i, s in enumerate(SUGGESTIONS):
    with cols[i]:
        if st.button(s, key=f"sug_{i}"):
            st.session_state.current_q = s
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

question = st.text_input(
    "question",
    value=st.session_state.current_q,
    placeholder="Ask anything about company policies…",
    label_visibility="collapsed",
)

ask_col, _ = st.columns([1, 3])
with ask_col:
    ask = st.button("⚡ Ask", key="ask_btn")

# ── Run query with streaming ──────────────────────────────────────────────────
if ask and question.strip():
    st.session_state.current_q = ""

    st.markdown("<div class='answer-label'>Answer</div>", unsafe_allow_html=True)
    answer_box = st.empty()

    full_answer = ""
    sources = []

    with st.spinner("Searching knowledge base…"):
        for token in stream_answer(question, history=st.session_state.history):
            if isinstance(token, dict):          # final sentinel with sources
                sources = token["sources"]
            else:
                full_answer += token
                answer_box.markdown(
                    f"<div class='answer-card'>{full_answer}▌</div>",
                    unsafe_allow_html=True,
                )

    answer_box.markdown(
        f"<div class='answer-card'>{full_answer}</div>",
        unsafe_allow_html=True,
    )

    # Save to history & log
    st.session_state.history.append({
        "question": question,
        "answer": full_answer,
        "sources": sources,
        "feedback": None,
    })
    log_query(question, full_answer, sources)

    # ── Feedback buttons ──────────────────────────────────────────────────────
    fb_col1, fb_col2, _ = st.columns([1, 1, 6])
    with fb_col1:
        if st.button("👍 Helpful", key="fb_up"):
            st.session_state.history[-1]["feedback"] = "👍"
            log_query(question, full_answer, sources, feedback="👍")
            st.toast("Thanks for the feedback!")
    with fb_col2:
        if st.button("👎 Not helpful", key="fb_down"):
            st.session_state.history[-1]["feedback"] = "👎"
            log_query(question, full_answer, sources, feedback="👎")
            st.toast("Thanks — we'll use this to improve.")

    # ── Retrieved chunks ──────────────────────────────────────────────────────
    if sources:
        with st.expander(f"🔍 View {len(sources)} retrieved chunks"):
            for i, src in enumerate(sources):
                st.markdown(
                    f"<div class='chunk-card'>"
                    f"<div class='chunk-header'>"
                    f"<span class='chunk-num'>Chunk {i+1}</span>"
                    f"<span>"
                    f"<span class='score-badge'>rerank {src.get('rerank_score', 0):.3f}</span>"
                    f"<span class='source-tag'>📄 {src['source']} p.{src['page']}</span>"
                    f"</span></div>"
                    f"{src['chunk']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No relevant chunks found above the relevance threshold.")

# ── Previous Q&A ──────────────────────────────────────────────────────────────
if len(st.session_state.history) > 1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.7rem;"
        "color:#3d4355;text-transform:uppercase;letter-spacing:0.1em;"
        "margin-bottom:12px;'>Previous in this session</div>",
        unsafe_allow_html=True,
    )
    for item in reversed(st.session_state.history[:-1]):
        fb = item.get("feedback", "")
        label = f"{'👍 ' if fb == '👍' else '👎 ' if fb == '👎' else ''}❯ {item['question']}"
        with st.expander(label):
            st.markdown(
                f"<div class='answer-card' style='margin:0'>{item['answer']}</div>",
                unsafe_allow_html=True,
            )
