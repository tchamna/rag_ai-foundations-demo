from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io, csv
from src.rag_pipeline import RAGPipeline, load_corpus, build_documents, VectorIndex
from src.config import VECTORSTORE_DIR, DATA_DIR, TOP_K, SIM_THRESHOLD, USE_RERANKER

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory transcript for this backend instance
TRANSCRIPT = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(
    question: str = Form(...),
    threshold: float = Form(SIM_THRESHOLD),
    use_chatgpt: bool = Form(False),
):
    q = question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        rag = RAGPipeline(VECTORSTORE_DIR, use_chatgpt=use_chatgpt, use_reranker=USE_RERANKER)
        rag.ensure_loaded()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Vectorstore not found. Build the index first.")

    retrieved = rag.retrieve(q, TOP_K)
    ctxs = [c for c in retrieved if c.get("score", 0.0) >= threshold]
    if not ctxs and retrieved:
        # fallback to top-k so generator has material
        ctxs = retrieved[:TOP_K]

    if not ctxs:
        result = {"answer": "I could not find this in the documents.", "prompt": "No contexts found.", "mode": "no_context"}
    else:
        result = rag.answer(q, ctxs)

    answer = result.get("answer", "")
    contexts = [{"source": c["meta"].get("source", "N/A"), "text": c["text"], "score": c.get("score")} for c in ctxs]

    TRANSCRIPT.append({"question": q, "answer": answer, "contexts": contexts})

    return {"answer": answer, "contexts": contexts, "prompt": result.get("prompt", ""), "mode": result.get("mode", "")}


@app.post("/rebuild")
async def rebuild():
    corpus = load_corpus(DATA_DIR)
    docs = build_documents(corpus)
    index = VectorIndex()
    index.build(docs)
    index.save(VECTORSTORE_DIR)
    return {"status": "ok", "chunks": len(docs)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    target_dir = DATA_DIR / "user_uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file.filename
    contents = await file.read()
    with open(target, "wb") as f:
        f.write(contents)
    return {"status": "saved", "name": file.filename}


@app.get("/transcript")
async def get_transcript():
    # Return CSV stream for download
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Q#", "Question", "Answer", "Source", "Chunk Text", "Score"])

    for i, item in enumerate(TRANSCRIPT, 1):
        q = item.get("question", "")
        a = item.get("answer", "")
        ctxs = item.get("contexts", [])
        if ctxs:
            for c in ctxs:
                writer.writerow([i, q, a, c.get("source", "N/A"), c.get("text", ""), c.get("score", "")])
        else:
            writer.writerow([i, q, a, "N/A", "N/A", ""])

    buffer.seek(0)
    headers = {"Content-Disposition": "attachment; filename=qa_transcript.csv"}
    return StreamingResponse(buffer, media_type="text/csv", headers=headers)


@app.get("/transcript/json")
async def get_transcript_json():
    return JSONResponse(TRANSCRIPT)
