import gradio as gr
from rag_pipeline import RAGPipeline
from config import VECTORSTORE_DIR, TOP_K, USE_RERANKER


def make_rag():
    rag = RAGPipeline(VECTORSTORE_DIR, use_chatgpt=False, use_reranker=USE_RERANKER)
    try:
        rag.ensure_loaded()
    except FileNotFoundError:
        raise
    return rag


rag = make_rag()


def qa_fn(question: str):
    q = question.strip()
    if not q:
        return "", ""
    retrieved = rag.retrieve(q, TOP_K)
    ctxs = retrieved or []
    if not ctxs:
        # lexical fallback
        docs = rag.index.docs
        idxs = [i for i, d in enumerate(docs) if q.lower() in d["text"].lower()]
        ctxs = [docs[i] for i in idxs[:TOP_K]]

    result = rag.answer(q, ctxs)
    answer = result.get("answer", "I could not find this in the documents.")
    ctx_text = "\n\n".join([f"[{i+1}] {c['meta']['source']}: {c['text']}" for i, c in enumerate(ctxs)])
    return answer, ctx_text


with gr.Blocks() as demo:
    gr.Markdown("# Banking & Fe'efe'e Assistant — Gradio Demo")
    with gr.Row():
        inp = gr.Textbox(label="Question", lines=2)
        btn = gr.Button("Ask")
    out_answer = gr.Textbox(label="Answer", lines=6)
    out_ctx = gr.Textbox(label="Retrieved Contexts", lines=8)
    btn.click(qa_fn, inputs=inp, outputs=[out_answer, out_ctx])


def main():
    demo.launch()


if __name__ == '__main__':
    main()
