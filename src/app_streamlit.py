import sys
from pathlib import Path
import os

# Ensure repository root is on sys.path so imports like `from src.rag_pipeline` work
# whether streamlit is launched from the repo root or from inside `src/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from src.rag_pipeline import RAGPipeline, load_corpus, build_documents, VectorIndex
from src.config import DATA_DIR, VECTORSTORE_DIR, TOP_K, SIM_THRESHOLD, USE_RERANKER
from src.theme import get_theme_css


# Cached RAG pipeline factory to avoid reinitializing models on every submit
@st.cache_resource
def get_rag_pipeline(use_chatgpt_flag: bool, use_reranker_flag: bool):
    rp = RAGPipeline(
        VECTORSTORE_DIR,
        use_chatgpt=use_chatgpt_flag,
        use_reranker=use_reranker_flag,
        refine_phrasebook_with_gpt=False,
    )
    rp.ensure_loaded()
    return rp


def get_rag_pipeline_safe(use_chatgpt_flag: bool, use_reranker_flag: bool):
    """Create the RAG pipeline but catch missing heavy deps and vectorstore errors.
    Returns None on failure and shows a friendly Streamlit warning where appropriate.
    """
    try:
        return get_rag_pipeline(use_chatgpt_flag, use_reranker_flag)
    except FileNotFoundError as e:
        st.warning("Vector store not found. Build the index (sidebar) or run the ingest script before querying.")
        return None
    except RuntimeError as e:
        # This commonly happens when sentence-transformers/transformers is missing
        st.warning("A required ML package is not installed in this runtime (sentence-transformers/transformers).\n"
                   "Use the minimal runtime image or enable the ML runtime that includes these packages.")
        return None
    except Exception as e:
        st.error(f"Failed to initialize retrieval pipeline: {e}")
        return None

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

# runtime override for top_k and reranker to speed up/slow down behavior
runtime_top_k = st.sidebar.slider("Top K (retrieval)", 1, 20, TOP_K)
runtime_use_reranker = st.sidebar.checkbox("Use reranker (may be slower)", value=USE_RERANKER)

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
    st.subheader("Chat — ask a question")

    # Prepare session state for chat messages and input
    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = ""
    if "transcript" not in st.session_state:
        st.session_state["transcript"] = []

    def handle_submit():
        user_text = st.session_state.get("chat_input", "").strip()
        if not user_text:
            return

        # Append user message immediately (chat UI)
        st.session_state["transcript"].append({"role": "user", "text": user_text})

        # clear input for next message
        st.session_state["chat_input"] = ""

        try:
            # Use cached pipeline to avoid reloading models each submit
            rag = get_rag_pipeline(use_chatgpt, runtime_use_reranker)
            with st.spinner("Running retrieval and generation..."):
                ctxs = rag.retrieve(user_text, runtime_top_k)
                result = rag.answer(user_text, ctxs)

            assistant_text = result.get("answer", "(no answer)")

            # Append assistant reply
            st.session_state["transcript"].append({"role": "assistant", "text": assistant_text, "contexts": ctxs, "prompt": result.get("prompt")})
        except FileNotFoundError as e:
            st.session_state["transcript"].append({"role": "assistant", "text": f"Error: {e}. Build the vector store first."})
        except Exception as e:
            st.session_state["transcript"].append({"role": "assistant", "text": f"Error during query: {e}"})

    # Chat message area (render transcript)
    chat_container = st.container()
    Assistant_name = "Shck Tchamna"
    with chat_container:
        for msg in st.session_state.get("transcript", []):
            if msg.get("role") == "user":
                st.markdown(f"**You:** {msg.get('text')}")
            else:
                st.markdown(f"**{Assistant_name}:** {msg.get('text')}" )

    # Input box that submits on Enter via on_change
    st.text_input("", key="chat_input", placeholder="Type a message and press Enter", on_change=handle_submit)

    # ----------------------ε
    # Transcript Download
    # -------------------------
    if st.session_state.get("transcript"):
        st.subheader("Download Q&A Transcript")

        rows = []

        # Support both legacy QA entries and role-based chat transcripts.
        # Build a list of QA pairs: (question_text, answer_text, contexts)
        qa_pairs = []

        # If transcript entries already have 'question' keys, use them directly
        legacy_entries = all(isinstance(it, dict) and ("question" in it or "answer" in it) for it in st.session_state.get("transcript", []))
        if legacy_entries:
            for item in st.session_state.get("transcript", []):
                q = item.get("question", "")
                a = item.get("answer", "")
                ctxs = item.get("contexts", [])
                qa_pairs.append((q, a, ctxs))
        else:
            # Walk through role-based messages and pair user -> next assistant
            msgs = st.session_state.get("transcript", [])
            i = 0
            while i < len(msgs):
                m = msgs[i]
                if m.get("role") == "user":
                    q_text = m.get("text", "")
                    # find next assistant message
                    a_text = ""
                    a_ctxs = []
                    j = i + 1
                    while j < len(msgs):
                        if msgs[j].get("role") == "assistant":
                            a_text = msgs[j].get("text", "")
                            a_ctxs = msgs[j].get("contexts", []) or msgs[j].get("ctxs", []) or []
                            break
                        j += 1
                    qa_pairs.append((q_text, a_text, a_ctxs))
                    i = j + 1
                else:
                    i += 1

        # Now expand qa_pairs into rows (one row per context, or a single N/A row)
        for idx, (q, a, ctxs) in enumerate(qa_pairs, 1):
            if ctxs:
                for ctx in ctxs:
                    raw = ctx.get("text", "N/A")
                    eng = None
                    feefee = None
                    fr = None
                    if "|" in raw:
                        parts = [p.strip() for p in raw.split("|")]
                        if len(parts) == 3:
                            eng, feefee, fr = parts
                            eng = _clean_phrase(eng)
                            feefee = _clean_phrase(feefee)
                            fr = _clean_phrase(fr)
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

                    cleaned_chunk = feefee or eng or raw

                    rows.append({
                        "Q#": idx,
                        "Question": q,
                        "Answer": a,
                        "Source": ctx.get("meta", {}).get("source", ctx.get("source", "N/A")),
                        "Chunk Text": cleaned_chunk,
                        "English": eng,
                        "Fe'efe'e": feefee,
                        "French": fr,
                        "Score": ctx.get("score", None)
                    })
            else:
                rows.append({
                    "Q#": idx,
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

# Determine the most recent user query from the transcript (fallback to 'introductions')
last_user = ""
for m in reversed(st.session_state.get("transcript", [])):
    if m.get("role") == "user":
        last_user = m.get("text", "")
        break
current_query = last_user.strip() if last_user and last_user.strip() else "introductions"

with col2:
    st.markdown("### Retrieved Chunks")
    try:
        # Use cached pipeline and runtime_top_k
        rag = get_rag_pipeline(use_chatgpt, runtime_use_reranker)
        ctxs = rag.retrieve(current_query, runtime_top_k)
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
