import streamlit as st
from pathlib import Path
import pandas as pd

from rag_pipeline import RAGPipeline, load_corpus, build_documents, VectorIndex
from config import DATA_DIR, VECTORSTORE_DIR, TOP_K

st.set_page_config(page_title="Banking & Fe'efe'e Assistant (RAG Demo)", layout="wide")

st.title("🏦 Banking & Fe'efe'e Assistant — RAG Demo")
st.caption("Local demo: FAISS + SentenceTransformers + FLAN-T5 | Upload statements and ask questions.")

# Sidebar — Build / Rebuild index
st.sidebar.header("Index Builder")
rebuild = st.sidebar.button("Rebuild Vector Store")
status = st.sidebar.empty()

def rebuild_index():
    status.info("Building vector store...")
    corpus = load_corpus(DATA_DIR)
    docs = build_documents(corpus)
    index = VectorIndex()
    index.build(docs)
    index.save(VECTORSTORE_DIR)
    status.success(f"Built vector store with {len(docs)} chunks.")

if rebuild:
    rebuild_index()

# Sidebar — Upload
st.sidebar.header("Upload Statement")
upload = st.sidebar.file_uploader("Upload .txt, .csv or .xlsx", type=["txt","csv","xlsx"])
if upload:
    target_dir = DATA_DIR / "sample_statements"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"user_{upload.name}"
    target.write_bytes(upload.getvalue())
    st.sidebar.success(f"Saved: {target.name}")
    st.sidebar.info("Click 'Rebuild Vector Store' to include this in retrieval.")

# Sidebar — Similarity Threshold
st.sidebar.header("Retrieval Settings")
threshold = st.sidebar.slider(
    "Similarity threshold (filter weak matches)", 
    min_value=0.0, max_value=1.0, value=0.25, step=0.05
)

# Save transcript
if "transcript" not in st.session_state:
    st.session_state["transcript"] = []

# Post-processor for citations
def enforce_citations(answer: str, contexts):
    if not contexts:
        return answer
    numbers = [f"[{i+1}]" for i in range(len(contexts))]
    if not any(n in answer for n in numbers):
        answer += " " + " ".join(numbers)
    return answer

# -------------------- Main Q&A --------------------
st.subheader("Ask a question")

with st.form(key="qa_form"):
    query = st.text_input(
        "e.g., What recurring charges do I have in May? How do I avoid overdraft fees?"
    )
    submitted = st.form_submit_button("Get Answer")

if submitted and query.strip():
    try:
        rag = RAGPipeline(VECTORSTORE_DIR)
        rag.ensure_loaded()

        # Retrieve and apply threshold filter
        raw_ctxs = rag.retrieve(query.strip(), TOP_K)
        ctxs = [c for c in raw_ctxs if c["score"] >= threshold]

        if not ctxs:
            fixed_answer = "I could not find this in the documents."
            result = {"prompt": f"No contexts >= {threshold}"}
        else:
            result = rag.answer(query.strip(), ctxs)
            fixed_answer = result["answer"]

        st.markdown("### Answer")
        st.write(fixed_answer)

        # Save transcript
        st.session_state["transcript"].append({
            "question": query.strip(),
            "answer": fixed_answer,
            "contexts": ctxs
        })

        if ctxs:
            with st.expander("Show prompt (debug)"):
                st.code(result["prompt"], language="text")

    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Use the sidebar to build the vector store first.")

# -------------------- Retrieved Chunks --------------------
st.subheader("Retrieved Chunks")
if query.strip():
    try:
        rag = RAGPipeline(VECTORSTORE_DIR)
        rag.ensure_loaded()
        ctxs = rag.retrieve(query.strip(), TOP_K)
        for i, c in enumerate(ctxs, 1):
            if c["score"] >= threshold:  # ✅ respect threshold
                with st.expander(f"🔎 Chunk {i} — {c['meta']['source']} (score={c['score']:.3f})", expanded=False):
                    parts = [p.strip() for p in c["text"].split("|")]
                    if len(parts) == 3:
                        st.markdown(f"**English:** {parts[0]}")
                        st.markdown(f"**Fè’éfě’è:** {parts[1]}")
                        st.markdown(f"**French:** {parts[2]}")
                    else:
                        st.text(c["text"])
    except Exception:
        st.caption("⚠️ Build the index to preview retrieved chunks.")

# -------------------- Transcript Download --------------------
if st.session_state.get("transcript"):
    st.subheader("Download Q&A Transcript")

    rows = []
    for i, item in enumerate(st.session_state["transcript"], 1):
        q = item.get("question", "")
        a = item.get("answer", "")
        ctxs = item.get("contexts", [])

        if ctxs:
            for ctx in ctxs:
                rows.append({
                    "Q#": i,
                    "Question": q,
                    "Answer": a,
                    "Source": ctx.get("source", "N/A"),
                    "Chunk Text": ctx.get("text", "N/A"),
                    "Score": ctx.get("score", None)
                })
        else:
            rows.append({
                "Q#": i,
                "Question": q,
                "Answer": a,
                "Source": "N/A",
                "Chunk Text": "N/A",
                "Score": None
            })

    df = pd.DataFrame(rows)
    csv = df.to_csv(index=False, encoding="utf-8-sig")

    st.download_button(
        "📥 Download Transcript (CSV)",
        csv,
        "qa_transcript.csv",
        "text/csv",
        key="download-csv"
    )
