import gradio as gr
from rag_pipeline import RAGPipeline, load_corpus, build_documents, VectorIndex
from src.config import DATA_DIR, VECTORSTORE_DIR, TOP_K, SIM_THRESHOLD, USE_RERANKER
import os
import io
import pandas as pd
import unicodedata
import re
import pickle
import tempfile


def _clean_phrase(s: str) -> str:
    if not s:
        return s
    out = s.strip()
    out = re.sub(r"^\s*\d+\)\s*", "", out)
    out = re.sub(r"^\s*\d+[:\-]\s*", "", out)
    out = re.sub(r"^English:\s*", "", out, flags=re.I)
    return out.strip()



# transcript stored in-memory for this Gradio session
TRANSCRIPT = []


def make_rag(use_chatgpt: bool = False):
    rag = RAGPipeline(VECTORSTORE_DIR, use_chatgpt=use_chatgpt, use_reranker=USE_RERANKER)
    rag.ensure_loaded()
    return rag


def rebuild_index():
    corpus = load_corpus(DATA_DIR)
    docs = build_documents(corpus)
    index = VectorIndex()
    index.build(docs)
    index.save(VECTORSTORE_DIR)
    return f"Built vector store with {len(docs)} chunks."


def handle_upload(uploaded_file):
    if not uploaded_file:
        return "No file uploaded."
    target_dir = DATA_DIR / "user_uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / uploaded_file.name
    # uploaded_file is a tempfile.SpooledTemporaryFile-like object in Gradio
    with open(target, "wb") as f:
        f.write(uploaded_file.read())
    return f"Saved: {target.name}. Run 'Rebuild Vector Store' to include it."


def qa_fn(question: str, threshold: float, use_chatgpt: bool):
    q = question.strip()
    if not q:
        return "", "", ""
    rag = make_rag(use_chatgpt=use_chatgpt)

    try:
        retrieved = rag.retrieve(q, TOP_K)
    except FileNotFoundError:
        return "", "", "Vectorstore not found. Run Rebuild."

    ctxs = [c for c in retrieved if c.get("score", 0.0) >= threshold]
    if not ctxs and retrieved:
        # fallback to top-k if threshold filtered everything
        ctxs = retrieved[:TOP_K]

    if not ctxs:
        result = {"prompt": "No contexts found.", "answer": "I could not find this in the documents.", "mode": "no_context"}
        answer = result["answer"]
        ctx_text = ""
        prompt = result["prompt"]
    else:
        result = rag.answer(q, ctxs)
        answer = result.get("answer", "I could not find this in the documents.")
        ctx_text = "\n\n".join([f"[{i+1}] {c['meta']['source']}: {c['text']}" for i, c in enumerate(ctxs)])
        prompt = result.get("prompt", "")

    # save to transcript
    TRANSCRIPT.append({
        "question": q,
        "answer": answer,
        "contexts": [{"source": c["meta"]["source"], "text": c["text"], "score": c.get("score")} for c in ctxs]
    })

    return answer, ctx_text, prompt


with gr.Blocks() as demo:
    gr.Markdown("# Banking Assistant — Gradio Demo")

    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                threshold_slider = gr.Slider(0.0, 1.0, value=SIM_THRESHOLD, label="Similarity threshold")
                chatgpt_chk = gr.Checkbox(label="Use ChatGPT API instead of FLAN-T5", value=False)
            ask_btn = gr.Button("Get Answer")
            out_answer = gr.Textbox(label="Answer", lines=6)
            out_ctx = gr.Textbox(label="Retrieved Contexts", lines=8)
            debug_prompt = gr.Textbox(label="Prompt (debug)", lines=8)

        with gr.Column(scale=1):
            rebuild_btn = gr.Button("Rebuild Vector Store")
            upload_file = gr.File(label="Upload .txt/.csv/.xlsx")
            upload_msg = gr.Textbox(label="Upload status")
            transcript_btn = gr.Button("Download Transcript (CSV)")
            browser_toggle = gr.Checkbox(label="Enable Vector DB Browser", value=False)

    ask_btn.click(qa_fn, inputs=[inp, threshold_slider, chatgpt_chk], outputs=[out_answer, out_ctx, debug_prompt])
    rebuild_btn.click(lambda: rebuild_index(), outputs=[upload_msg])
    upload_file.upload(fn=handle_upload, inputs=upload_file, outputs=[upload_msg])

    # Transcript download handler
    def download_transcript():
        if not TRANSCRIPT:
            return None
        rows = []
        for i, item in enumerate(TRANSCRIPT, 1):
            q = item.get("question", "")
            a = item.get("answer", "")
            ctxs = item.get("contexts", [])
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

        def _nfc(v):
            if v is None:
                return ""
            return unicodedata.normalize("NFC", str(v))

        for r in rows:
            for k in ("Question", "Answer", "Source", "Chunk Text"):
                r[k] = _nfc(r.get(k, ""))

        df = pd.DataFrame(rows)
        csv_text = df.to_csv(index=False)
        return csv_text

    transcript_btn.click(lambda: download_transcript(), outputs=[out_answer])


def main():
    demo.launch()


if __name__ == '__main__':
    main()
