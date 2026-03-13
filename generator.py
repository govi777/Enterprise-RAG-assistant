import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve

load_dotenv()

# ── Startup validation ────────────────────────────────────────────────────────
def _get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "your_actual_api_key_here":
        raise EnvironmentError(
            "OPENAI_API_KEY is missing or still set to the placeholder value.\n"
            "Please add your real key to the .env file:\n"
            "  OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(question, retrieved_chunks, history=None):
    context = "\n\n".join(
        [f"[{c.get('source', 'company_docs.pdf')} p.{c.get('page', '—')}]\n{c.get('chunk', c) if isinstance(c, dict) else c}"
         for c in retrieved_chunks]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an enterprise assistant. "
                "Answer ONLY using the provided context. "
                "Always cite the source and page number when relevant (e.g. 'According to company_docs.pdf p.3'). "
                "If the answer is not in the context, say \"I don't know.\""
            ),
        }
    ]

    # Inject conversation history (last N turns for memory)
    if history:
        for turn in history[-4:]:          # keep last 4 turns as context
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    return messages


# ── Non-streaming answer ──────────────────────────────────────────────────────
def generate_answer(question, history=None):
    client = _get_client()
    retrieved = retrieve(question)

    if not retrieved:
        return "I couldn't find any relevant information in the documents.", []

    messages = build_prompt(question, retrieved, history=history)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    answer = response.choices[0].message.content
    return answer, retrieved


# ── Streaming answer (generator) ─────────────────────────────────────────────
def stream_answer(question, history=None):
    """
    Yields text tokens one by one for streaming UI.
    Usage:
        for token in stream_answer(question, history):
            print(token, end="", flush=True)
    Also returns retrieved chunks via a sentinel at the end:
        yields {"sources": [...]} as the last item
    """
    client = _get_client()
    retrieved = retrieve(question)

    if not retrieved:
        yield "I couldn't find any relevant information in the documents."
        yield {"sources": []}
        return

    messages = build_prompt(question, retrieved, history=history)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

    yield {"sources": retrieved}


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    question = input("Ask a question: ")
    answer, sources = generate_answer(question)
    print("\nAnswer:\n", answer)
    print("\nSources:")
    for s in sources:
        print(f"  [{s['source']} p.{s['page']}] rerank={s['rerank_score']:.3f}")
        print(f"  {s['chunk'][:120]}...")
