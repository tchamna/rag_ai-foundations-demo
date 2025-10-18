import streamlit as st
from pathlib import Path
import pandas as pd
import pickle
import io
import unicodedata
import re


def _clean_phrase(s: str) -> str:
    if not s:
        return s
    out = s.strip()
    # remove leading numeric markers like '520)' or '520) '
    out = re.sub(r"^\s*\d+\)\s*", "", out)
    out = re.sub(r"^\s*\d+[:\-]\s*", "", out)
    # remove leading label like 'English:' if present
    out = re.sub(r"^English:\s*", "", out, flags=re.I)
    return out.strip()

from rag_pipeline import RAGPipeline, load_corpus, build_documents, VectorIndex
from src.config import DATA_DIR, VECTORSTORE_DIR, TOP_K, SIM_THRESHOLD, USE_RERANKER
from src.theme import get_theme_css

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="Banking Assistant (RAG Demo)", layout="wide")
st.title("🏦 Banking Assistant — RAG Demo")
st.caption("Demo with FAISS + SentenceTransformers + FLAN-T5 or ChatGPT | Upload docs and ask questions.")

# -------------------------
# Session State
# -------------------------
if "transcript" not in st.session_state:
    st.session_state["transcript"] = []
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

# -------------------------
# Theme Toggle
# -------------------------
if st.sidebar.button("Toggle Dark Mode"):
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]
    st.rerun()

# Apply theme via centralized theme provider
if st.session_state["dark_mode"]:
    st.markdown(get_theme_css(True), unsafe_allow_html=True)
else:
    st.markdown(get_theme_css(False), unsafe_allow_html=True)

# -------------------------
# Sidebar — Index Builder
# -------------------------
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

# -------------------------
# Sidebar — Upload
# -------------------------
st.sidebar.header("Upload Document")
upload = st.sidebar.file_uploader("Upload .txt, .csv or .xlsx", type=["txt","csv","xlsx"])
if upload:
    target_dir = DATA_DIR / "user_uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / upload.name
    target.write_bytes(upload.getvalue())
    st.sidebar.success(f"Saved: {target.name}")
    st.sidebar.info("Click 'Rebuild Vector Store' to include this in retrieval.")

# -------------------------
# Sidebar — Retrieval Settings
# -------------------------
st.sidebar.header("Retrieval Settings")
threshold = st.sidebar.slider(
    "Similarity threshold", 0.0, 1.0, SIM_THRESHOLD, 0.05
)

# -------------------------
# Sidebar — Choose Generator
# -------------------------
st.sidebar.header("Answer Generator")
use_chatgpt = st.sidebar.checkbox("Use ChatGPT API instead of FLAN-T5", value=False)

# -------------------------
# Post-processor for citations
# -------------------------
def enforce_citations(answer: str, contexts):
    if not contexts:
        return answer
    numbers = [f"[{i+1}]" for i in range(len(contexts))]
    if not any(n in answer for n in numbers):
        answer += " " + " ".join(numbers)
    return answer

# -------------------------
# Main — Q&A
# -------------------------
# Two-column layout: main Q&A (col1) + retrieval preview (col2)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask a question")
    query = st.text_input("e.g., What is the ATM withdrawal limit? How to say 'I love you' in Fe'efe'e?", key="query_input")

    # Query handling
    if query and query.strip():
        try:
            rag = RAGPipeline(
                VECTORSTORE_DIR,
                use_chatgpt=use_chatgpt,
                use_reranker=USE_RERANKER,
                refine_phrasebook_with_gpt=False,
            )

            rag.ensure_loaded()
            ctxs = rag.retrieve(query.strip(), TOP_K)
            result = rag.answer(query.strip(), ctxs)

            mode = result.get("mode", "")
            if mode in ("gpt", "phrasebook+gpt"):
                st.info("⚡ Using ChatGPT API for generation")
            elif mode == "flan":
                st.info("🧠 Using local FLAN-T5 model")
            elif mode == "faq":
                st.success("✅ Direct FAQ match (no generator)")
            elif mode == "phrasebook":
                st.success("✅ Direct phrasebook match (no generator)")
            elif mode == "fallback":
                st.warning("↩️ Fallback to top chunk text")

            fixed_answer = result["answer"]

            st.markdown("### Answer")
            st.write(fixed_answer)

            # Save transcript with contexts
            st.session_state["transcript"].append({
                "question": query.strip(),
                "answer": fixed_answer,
                "contexts": [
                    {
                        "source": c["meta"]["source"],
                        "text": c["text"],
                        "score": c.get("score", None)
                    }
                    for c in ctxs
                ]
            })

            st.session_state["last_query"] = query.strip()

            if ctxs:
                with st.expander("Show prompt (debug)"):
                    st.code(result.get("prompt", ""), language="text")

        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Use the sidebar to build the vector store first.")
        except Exception as e:
            st.error(f"Error during query: {e}")

    # -------------------------
    # Transcript Download
    # -------------------------
    if st.session_state.get("transcript"):
        st.subheader("Download Q&A Transcript")

        rows = []
        for i, item in enumerate(st.session_state["transcript"], 1):
            q = item.get("question", "")
            a = item.get("answer", "")
            ctxs = item.get("contexts", [])

            if ctxs:
                for ctx in ctxs:
                    raw = ctx.get("text", "N/A")
                    # try to parse phrasebook style: English / Fe'efe'e / French
                    eng = None
                    feefee = None
                    fr = None
                    # first try the pipe-delimited style
                    if "|" in raw:
                        parts = [p.strip() for p in raw.split("|")]
                        if len(parts) == 3:
                            eng, feefee, fr = parts
                            eng = _clean_phrase(eng)
                            feefee = _clean_phrase(feefee)
                            fr = _clean_phrase(fr)
                    # fallback to labeled format
                    if (not eng) and "Fe'efe'e:" in raw:
                        try:
                            if "English:" in raw:
                                eng = raw.split("English:", 1)[1].split("Fe'efe'e:", 1)[0].strip()
                                eng = _clean_phrase(eng)
                            if "Fe'efe'e:" in raw:
                                feefee = raw.split("Fe'efe'e:", 1)[1]
                                if "French:" in feefee:
                                    feefee, fr_part = feefee.split("French:", 1)
                                    fr = _clean_phrase(fr_part)
                                feefee = _clean_phrase(feefee)
                        except Exception:
                            pass

                    # final cleaned chunk text (prefer feefee then english then raw)
                    cleaned_chunk = feefee or eng or raw

                    rows.append({
                        "Q#": i,
                        "Question": q,
                        "Answer": a,
                        "Source": ctx.get("source", "N/A"),
                        "Chunk Text": cleaned_chunk,
                        "English": eng,
                        "Fe'efe'e": feefee,
                        "French": fr,
                        "Score": ctx.get("score", None)
                    })
            else:
                rows.append({
                    "Q#": i,
                    "Question": q,
                    "Answer": a,
                    "Source": "N/A",
                    "Chunk Text": "N/A",
                    "English": None,
                    "Fe'efe'e": None,
                    "French": None,
                    "Score": None
                })

        # Normalize unicode to NFC so characters render correctly in Excel/
        # other viewers. Convert the CSV to UTF-8 with BOM bytes for
        # best compatibility with Microsoft Excel on Windows.
        def _nfc(v):
            if v is None:
                return ""
            return unicodedata.normalize("NFC", str(v))

        # Apply normalization
        for r in rows:
            for k in ("Question", "Answer", "Source", "Chunk Text"):
                r[k] = _nfc(r.get(k, ""))

        df = pd.DataFrame(rows)
        # Only provide an XLSX download which usually avoids encoding/locale issues
        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="transcript")
            xlsx_data = output.getvalue()

        st.download_button(
            "📥 Download Transcript (XLSX)",
            data=xlsx_data,
            file_name="qa_transcript.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download-xlsx",
        )

    # (duplicate transcript block removed)

with col2:
    st.markdown("### Retrieved Chunks")
    try:
        # rag = RAGPipeline(VECTORSTORE_DIR, use_chatgpt=use_chatgpt)
        rag = RAGPipeline(
            VECTORSTORE_DIR,
            use_chatgpt=use_chatgpt,
            use_reranker=USE_RERANKER,
            refine_phrasebook_with_gpt=False, # disable GPT polishing of phrasebook hits
        )

        rag.ensure_loaded()
        ctxs = rag.retrieve(query if query.strip() else "introductions", TOP_K)
        for i, c in enumerate(ctxs, 1):
            with st.expander(
                f"🔎 Chunk {i} — {c['meta']['source']} (score={c['score']:.3f})",
                expanded=False
            ):
                parts = [p.strip() for p in c["text"].split("|")]
                if len(parts) == 3:
                    st.markdown(f"**English:** {parts[0]}")
                    st.markdown(f"**Fè’éfě’è:** {parts[1]}")
                    st.markdown(f"**French:** {parts[2]}")
                else:
                    st.text(c["text"])
    except Exception:
        st.caption("⚠️ Build the index to preview retrieved chunks.")

# -------------------------
# Vector DB Browser (moved into main layout on the right column)
# -------------------------
with col2:
    st.markdown("### Vector DB Browser")
    try:
        with open(VECTORSTORE_DIR / "docs.pkl", "rb") as f:
            docs = pickle.load(f)
        st.success(f"✅ Loaded {len(docs)} chunks")

        search_term = st.text_input("Search chunks (optional)", key="vector_search")
        if search_term:
            filtered_docs = [d for d in docs if search_term.lower() in d["text"].lower()]
            st.write(f"Found {len(filtered_docs)} matching chunks")
        else:
            filtered_docs = docs

        max_show = st.slider("How many to display?", 5, 50, 10, key="vector_max_show")

        for i, d in enumerate(filtered_docs[:max_show]):
            with st.expander(f"Chunk {i+1} — {d['meta'].get('source','N/A')}"):
                st.text(d["text"])

    except FileNotFoundError:
        st.warning("No docs found. Build the vector store to enable browsing.")
        if st.button("Build vector store now", key="vector_rebuild"):
            try:
                rebuild_index()
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Build failed: {e}")
    except Exception as e:
        st.error(f"Could not load docs.pkl: {e}")
