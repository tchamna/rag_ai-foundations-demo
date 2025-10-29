import sys
from pathlib import Path
import os

# Ensure repository root is on sys.path so imports like `from src.rag_pipeline` work
# whether streamlit is launched from the repo root or from inside `src/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
import os

# If Azure started Streamlit by running `streamlit run src/app_streamlit.py`
# directly, `st.set_page_config` may not have been called first. Guard a
# fallback here so the page title is still set when the environment variable
# `STREAMLIT_PAGE_CONFIG_SET` is not present. We make this a no-op when the
# env var is set to avoid duplicate set_page_config calls.
if not os.environ.get("STREAMLIT_PAGE_CONFIG_SET"):
    try:
        st.set_page_config(page_title="bank-rag-ai-app", page_icon="🏦", layout="wide")
        # Avoid re-running this block in the same process
        os.environ["STREAMLIT_PAGE_CONFIG_SET"] = "1"
    except Exception:
        # If set_page_config fails (e.g., called too late), continue silently.
        pass
import logging
import traceback
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


def _log_exception(e: Exception, ctx: str = ""):
    """Log exception details to stderr and an on-disk runtime log so Azure's container logs capture it."""
    msg = f"Exception in {ctx}: {e}\n" + traceback.format_exc()
    try:
        # Write to the system temp directory to avoid touching files in the
        # app directory (which can trigger App Service file-change restarts).
        import tempfile
        runtime_log = Path(tempfile.gettempdir()) / "rag_runtime_errors.log"
        with open(runtime_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    # Also send to stderr (visible in container logs)
    logging.error(msg)

# -------------------------
# Page Setup
# -------------------------
# Page config is handled in app.py to ensure set_page_config() is the
# first Streamlit call in the process. Here we only set the title/caption.
st.title("🏦 Banking Assistant — RAG Demo")
st.caption("Demo with FAISS + SentenceTransformers + FLAN-T5 or ChatGPT | Upload docs and ask questions.")

# -------------------------
# Runtime health banner
# -------------------------
def _check_runtime_health():
    """Return a tuple (vectorstore_ok: bool, sentence_transformers_available: bool).
    This is intentionally lightweight: it checks for the presence of the
    precomputed FAISS files and attempts a minimal import of
    sentence_transformers without loading models.
    """
    vs_ok = False
    try:
        vs_path = Path(VECTORSTORE_DIR)
        vs_ok = (vs_path / "faiss.index").exists() and (vs_path / "docs.pkl").exists()
    except Exception:
        vs_ok = False

    st_ok = False
    try:
        # Import the package only to detect presence, do not instantiate models.
        import sentence_transformers  # type: ignore
        st_ok = True
    except Exception:
        st_ok = False

    return vs_ok, st_ok


_vs_ok, _st_ok = _check_runtime_health()

# Transient runtime banner: show it once per session for a short duration
# using a temporary slot (`st.empty()`), then clear that slot so the
# message is removed without invoking an experimental rerun which can
# sometimes interact badly with Azure deployments.
if 'runtime_banner_hidden' not in st.session_state:
    st.session_state['runtime_banner_hidden'] = False

if not st.session_state['runtime_banner_hidden']:
    slot = st.empty()
    with slot.container():
        col_a, col_b = st.columns([3, 1])
        with col_a:
            if _vs_ok:
                st.success("Vectorstore: precomputed index found (faiss.index + docs.pkl)")
            else:
                st.error("Vectorstore: NOT found. Run `python src/precompute_embeddings.py` and commit `vectorstore/` or rebuild index via sidebar.")
        with col_b:
            if _st_ok:
                st.success("sentence-transformers: installed")
            else:
                st.info("sentence-transformers: NOT installed — lexical fallback / prebuilt vectors will be used")

    try:
        import time
        # Delay briefly so the user notices the banner (short to avoid long page freezes)
        time.sleep(4)
        # Clear the slot (removes the banner from the current render)
        slot.empty()
        # Mark hidden so it won't be shown again this session
        st.session_state['runtime_banner_hidden'] = True
    except Exception:
        # If anything goes wrong don't crash; leave banner visible
        pass

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
if st.sidebar.button("Toggle Bright Mode"):
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
use_chatgpt = st.sidebar.checkbox(
    "Use ChatGPT API instead of FLAN-T5 (disabled)",
    value=False,
    disabled=True,
    help="ChatGPT is disabled in this runtime to avoid external API calls. Enable manually in code if needed.",
)

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

        # Defensive dedupe: if the last user message is identical to the
        # current input, do not append again. This avoids double-posting when
        # both the input's on_change and a button call this handler during
        # the same rerun.
        last_msgs = st.session_state.get("transcript", [])
        if last_msgs:
            last = last_msgs[-1]
            if last.get("role") == "user" and (last.get("text", "") or "") == user_text:
                # Clear the input and exit early
                st.session_state["chat_input"] = ""
                return

        # Append user message immediately (chat UI)
        st.session_state["transcript"].append({"role": "user", "text": user_text})

        # clear input for next message
        st.session_state["chat_input"] = ""

        try:
            # Use safe pipeline factory which shows friendly warnings when deps
            # or vectorstore are missing instead of raising uncaught exceptions
            # Allow a per-submit override so the 'Ask with ChatGPT' button
            # can request ChatGPT for a single query without changing the
            # sidebar setting permanently.
            override = st.session_state.pop("chat_use_chatgpt_override", None)
            local_use_chatgpt = use_chatgpt if override is None else bool(override)

            rag = get_rag_pipeline_safe(local_use_chatgpt, runtime_use_reranker)
            if rag is None:
                st.session_state["transcript"].append({"role": "assistant", "text": "Error: retrieval pipeline not available in this runtime. Check logs or enable the ML runtime."})
                return

            # Do retrieval and generation in guarded blocks so any exception
            # (including unusual BaseException subclasses) is logged and
            # converted into a friendly message rather than letting Streamlit
            # crash or the process die silently in production.
            with st.spinner("Running retrieval..."):
                try:
                    ctxs = rag.retrieve(user_text, runtime_top_k)
                except FileNotFoundError as e:
                    st.session_state["transcript"].append({"role": "assistant", "text": f"Error: {e}. Build the vector store first."})
                    _log_exception(e, ctx="handle_submit - retrieve - FileNotFoundError")
                    return
                except Exception as e:
                    _log_exception(e, ctx="handle_submit - retrieve")
                    st.session_state["transcript"].append({"role": "assistant", "text": f"Retrieval failed: {e}. Check logs."})
                    return

            with st.spinner("Generating answer..."):
                try:
                    result = rag.answer(user_text, ctxs)
                except Exception as e:
                    # Generators may raise unexpected errors (OOM, missing libs);
                    # capture everything and log to help diagnose on Azure.
                    _log_exception(e, ctx="handle_submit - generate")
                    st.session_state["transcript"].append({"role": "assistant", "text": f"Generation failed: {e}. Check logs."})
                    return

            assistant_text = result.get("answer", "(no answer)")

            # Append assistant reply
            st.session_state["transcript"].append({"role": "assistant", "text": assistant_text, "contexts": ctxs, "prompt": result.get("prompt")})
        except BaseException as e:
            # Catch BaseException to try to surface even unusual errors (e.g., SystemError)
            # Log as much detail as possible to the runtime temp log so we can inspect
            # it from Azure Kudu if the process doesn't produce standard logs.
            _log_exception(e, ctx="handle_submit - base")
            try:
                st.session_state["transcript"].append({"role": "assistant", "text": f"Unexpected error during query: {e}. Check runtime logs."})
            except Exception:
                # If Streamlit is in a bad state, re-raise after logging so at least
                # the trace is persisted to disk for remote diagnosis.
                raise

    # Chat message area (render transcript)
    chat_container = st.container()
    Assistant_name = "Shck Tchamna"
    # Render chat messages inside bordered boxes to delimit question/answer
    # Safely escape user content to avoid HTML injection while allowing simple markdown.
    def _escape(s: str) -> str:
        import html
        return html.escape(s)

    user_box_style = """
    <div style='border-radius:8px; padding:10px; margin:6px 0; background:#e8f0ff; border:1px solid #c7ddff; text-align:left; color:#000;'>
    <strong style='color:#000;'>You:</strong>
    <div style='margin-top:6px; white-space:pre-wrap; color:#000;'>%s</div>
    </div>
    """

    assistant_box_style = """
    <div style='border-radius:8px; padding:10px; margin:6px 0; background:#f6fff0; border:1px solid #dff7d1; text-align:left; color:#000;'>
    <strong style='color:#000;'>%s:</strong>
    <div style='margin-top:6px; white-space:pre-wrap; color:#000;'>%s</div>
    </div>
    """

    with chat_container:
        for msg in st.session_state.get("transcript", []):
            try:
                role = msg.get("role")
                text = msg.get("text", "") or ""
                safe_text = _escape(text)
                if role == "user":
                    st.markdown(user_box_style % (safe_text,), unsafe_allow_html=True)
                else:
                    # assistant may include citations or other small HTML from generation; escape to be safe
                    name = Assistant_name
                    st.markdown(assistant_box_style % (name, safe_text), unsafe_allow_html=True)
            except Exception as e:
                _log_exception(e, ctx="render_chat_message")
                st.markdown(f"**{Assistant_name}:** (failed to render message)")

    # Input box that submits on Enter via on_change
    # Provide a collapsed (hidden) label to avoid Streamlit's empty-label
    # accessibility warning which may become an exception in future versions.
    # Single-line input with inline buttons in the same row to mimic ChatGPT
    input_col, btn_col = st.columns([8, 2])
    with input_col:
        st.text_input("Chat input", key="chat_input", placeholder="Type a message and press Enter", on_change=handle_submit, label_visibility="collapsed")

    # Helper to submit with optional ChatGPT override
    def submit_with_flag(flag=None):
        if flag is None:
            st.session_state.pop("chat_use_chatgpt_override", None)
        else:
            st.session_state["chat_use_chatgpt_override"] = flag
        handle_submit()

    with btn_col:
        # Place Send and ChatGPT buttons inline next to the input field
        # Disabled by default — enable when you want interactive sending
        if st.button("➤", disabled=True):
            submit_with_flag(None)
        if st.button("🤖", disabled=True):
            submit_with_flag(True)

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
        # Try XLSX export (preferred). If the runtime lacks openpyxl or
        # the stdlib 'xml' package (which openpyxl depends on), fall back
        # to a safe CSV export so the app does not crash on import/runtime.
        try:
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
        except (ModuleNotFoundError, ImportError) as e:
            # Common in constrained runtimes where openpyxl or stdlib xml
            # are missing. Log the full traceback and offer a CSV fallback.
            _log_exception(e, ctx="transcript_export_xlsx_import")
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.warning("XLSX export unavailable in this runtime; providing CSV fallback.")
            st.download_button(
                "📥 Download Transcript (CSV)",
                data=csv_bytes,
                file_name="qa_transcript.csv",
                mime="text/csv",
                key="download-csv",
            )
        except Exception as e:
            _log_exception(e, ctx="transcript_export")
            st.error(f"Failed to prepare transcript export: {e}")

    # (duplicate transcript block removed)

# Determine the most recent user query from the transcript (fallback to 'introductions')
last_user = ""
for m in reversed(st.session_state.get("transcript", [])):
    if m.get("role") == "user":
        last_user = m.get("text", "")
        break
# Only set current_query when we actually have a user query. When there is no
# prior user message we keep `current_query` as None so the preview area does
# not run retrieval and show results prematurely.
current_query = last_user.strip() if last_user and last_user.strip() else None

with col2:
    st.markdown("### Retrieved Chunks")
    # Optional debug toggle to surface reranker/score values
    show_debug_scores = st.checkbox("Show debug scores (rerank / score)", value=False, key="show_debug_scores")
    try:
        # If there's no user query yet, don't run retrieval — show an
        # instructional caption instead.
        if not current_query:
            st.caption("No query yet — ask a question in the chat to preview retrieved chunks.")
            ctxs = []
        else:
            # Use the safe pipeline factory here so missing deps/vectorstore
            # don't raise uncaught exceptions that would crash the app on Azure.
            rag = get_rag_pipeline_safe(use_chatgpt, runtime_use_reranker)
            if rag is None:
                st.caption("⚠️ Retrieval pipeline unavailable in this runtime (check logs).")
                ctxs = []
            else:
                try:
                    ctxs = rag.retrieve(current_query, runtime_top_k)
                except Exception as e:
                    _log_exception(e, ctx="retrieval_preview - retrieve")
                    st.caption("⚠️ Retrieval failed (check logs).")
                    ctxs = []

        for i, c in enumerate(ctxs, 1):
            try:
                header = f"🔎 Chunk {i} — {c['meta']['source']} (score={c['score']:.3f})"
                if show_debug_scores:
                    header += f" | final={c.get('final_score')} | rerank={c.get('rerank_score')}"

                with st.expander(header, expanded=False):
                    parts = [p.strip() for p in c["text"].split("|")]
                    if len(parts) == 3:
                        st.markdown(f"**English:** {parts[0]}")
                        st.markdown(f"**Fè’éfě’è:** {parts[1]}")
                        st.markdown(f"**French:** {parts[2]}")
                    else:
                        st.text(c["text"])
            except Exception as e:
                _log_exception(e, ctx=f"retrieval_preview - render chunk {i}")
                st.text("(failed to render chunk; check logs)")
    except Exception as e:
        _log_exception(e, ctx="retrieval_preview - outer")
        st.caption("⚠️ Build the index to preview retrieved chunks (check logs for errors).")

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
        show_all = st.checkbox("Show all chunks", value=False, key="vector_show_all")

        # Only show docs when the user searches OR explicitly requests to
        # show all. This avoids cluttering the UI with the entire corpus on
        # first load or when the user hasn't asked anything yet.
        if not search_term and not show_all:
            st.caption("No search term — enter text above or enable 'Show all chunks' to list stored chunks.")
            filtered_docs = []
        else:
            if search_term:
                filtered_docs = [d for d in docs if search_term.lower() in d["text"].lower()]
                st.write(f"Found {len(filtered_docs)} matching chunks")
            else:
                filtered_docs = docs

        max_show = st.slider("How many to display?", 5, 50, 10, key="vector_max_show")

        for i, d in enumerate(filtered_docs[:max_show]):
                with st.expander(f"Chunk {i+1} — {d['meta'].get('source','N/A')}"):
                    st.code(d["text"], language=None)
    except FileNotFoundError:
        st.warning("No docs found. Build the vector store to enable browsing.")
        if st.button("Build vector store now", key="vector_rebuild"):
            try:
                rebuild_index()
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Build failed: {e}")
    except Exception as e:
        _log_exception(e, ctx="vector_db_browser")
        st.error(f"Could not load docs.pkl: {e}")


    # (no top-level except here; inner try/except blocks handle page sections)
