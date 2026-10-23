from __future__ import annotations

from pathlib import Path
from typing import Literal

import asyncio
import json
import os
import re
import secrets
from fastapi import FastAPI, Request, HTTPException
from starlette.concurrency import run_in_threadpool
from fastapi import File, UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .security import GuardrailMiddleware
from .rag.config import env_sources_for_status
from .rag.cache import ResponseCache
from .rag.index import RagIndex
from .rag.llm_questions import suggest_questions
from .rag.intent import interpret_question
from .rag.model_comparison import compare_answers
from .rag.loaders import SUPPORTED_EXTENSIONS, load_documents
from .rag.index import chunk_documents
from .rag.synthesis import (
    ENV_SOURCES,
    compare_free_models,
    compare_models,
    detect_language,
    synthesize_answer,
    synthesis_status,
    translate_query_for_retrieval,
)


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

app = FastAPI(title="RAG Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# A random HttpOnly cookie is the capability for a browser's document workspace.
# Never use a caller-supplied path as a storage location.
workspace_locks: dict[str, asyncio.Lock] = {}

@app.middleware("http")
async def workspace_context(request: Request, call_next):
    if not (request.url.path.startswith("/api/generic") or request.url.path in {"/", "/generic", "/generic/compare"}):
        return await call_next(request)
    if request.method not in {"GET", "HEAD"}:
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
    token = request.cookies.get("rag_workspace", "")
    if not re.fullmatch(r"[a-f0-9]{64}", token):
        token = secrets.token_hex(32)
    request.state.workspace = ROOT / "storage" / "workspaces" / token
    lock = workspace_locks.setdefault(token, asyncio.Lock())
    async with lock:
        response = await call_next(request)
    response.set_cookie("rag_workspace", token, httponly=True, samesite="strict",
                        secure=os.getenv("RAG_SECURE_COOKIES") == "1", max_age=60*60*24*30)
    response.headers["Cache-Control"] = "no-store"
    return response


app.add_middleware(GuardrailMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[host.strip() for host in os.getenv("RAG_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",") if host.strip()])

def workspace_resources(request: Request):
    root = request.state.workspace
    index = RagIndex(root / "documents", root / "index.pkl")
    index.load_or_build()
    return index, ResponseCache(root / "responses.json")


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, pattern=r"\S")
    top_k: int = Field(default=8, ge=1, le=12)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=10)


class CompareRequest(ChatRequest):
    openai_models: list[Literal["gpt-5-nano", "gpt-6-luna", "gpt-6.1-sol"]] | None = Field(default=None, min_length=2, max_length=3)
    providers: list[Literal["openai", "groq", "claude", "ollama"]] | None = Field(default=None, max_length=4)
    models: list[str] | None = Field(default=None, max_length=3)




@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "generic.html")




@app.get("/generic")
def generic_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "generic.html")


@app.get("/generic/compare")
def generic_compare_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "generic_compare.html")














@app.get("/api/generic/status")
def generic_status(request: Request) -> dict:
    generic_rag, _ = workspace_resources(request)
    stats = generic_rag.stats()
    stats["env_sources"] = env_sources_for_status(ENV_SOURCES)
    stats["synthesis"] = synthesis_status()
    return stats


@app.get("/api/generic/questions")
def generic_questions(request: Request, batch: int = 0) -> dict:
    generic_rag, _ = workspace_resources(request)
    return suggest_questions(generic_rag, ResponseCache(request.state.workspace / "questions.json"), batch=batch)


@app.post("/api/generic/reindex")
def generic_reindex(request: Request) -> dict:
    generic_rag, _ = workspace_resources(request)
    generic_rag.rebuild()
    return generic_rag.stats()


@app.post("/api/generic/upload")
async def generic_upload(request: Request, files: list[UploadFile] = File(...)) -> dict:
    if len(files) > 10:
        raise HTTPException(400, "Upload at most 10 files at a time.")
    return await run_in_threadpool(process_workspace_upload, request, files)


@app.post("/api/generic/upload/stream")
async def stream_workspace_upload(request: Request, files: list[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(400, "Upload at most 10 files at a time.")
    async def events():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        def report(stage, detail):
            loop.call_soon_threadsafe(queue.put_nowait, {"stage":stage, "detail":detail})
        async def work():
            try:
                result = await run_in_threadpool(process_workspace_upload, request, files, report)
                await queue.put({"stage":"indexed", "result":result})
            except Exception:
                await queue.put({"stage":"error", "detail":"Document processing failed. Please try again."})
            finally:
                await queue.put(None)
        task = asyncio.create_task(work())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield json.dumps(event) + "\n"
        finally:
            await task
    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control":"no-store", "X-Accel-Buffering":"no"})


def process_workspace_upload(request: Request, files: list[UploadFile], progress=None) -> dict:
    index, _ = workspace_resources(request)
    upload_dir = index.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for file in files:
        name = Path((file.filename or "document").replace("\\", "/")).name
        target = None
        try:
            if (len(name) > 180 or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS
                    or any(ord(c) < 32 for c in name) or any(c in name for c in '<>:"|?*')
                    or name.split(".",1)[0].rstrip(" ").upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}):
                raise ValueError("Unsupported file type or filename too long.")
            if progress:
                progress("extracting", f"Reading {name}")
            content = file.file.read(20 * 1024 * 1024 + 1)
            if not content or len(content) > 20 * 1024 * 1024:
                raise ValueError("Files must be nonempty and no larger than 20 MB.")
            total_storage = sum(p.stat().st_size for p in (ROOT / "storage" / "workspaces").rglob("documents/uploads/*") if p.is_file())
            if total_storage + len(content) > 2 * 1024 * 1024 * 1024:
                raise ValueError("Server document storage is full. Contact the administrator.")
            if len(list(upload_dir.iterdir())) >= 100:
                raise ValueError("Workspace limit of 100 documents reached.")
            if sum(p.stat().st_size for p in upload_dir.iterdir()) + len(content) > 100 * 1024 * 1024:
                raise ValueError("Workspace storage limit of 100 MB reached. Remove a document first.")
            # Limit decompressed Office archives before passing them to parsers.
            if Path(name).suffix.lower() in {".docx", ".xlsx", ".odt"}:
                import io, zipfile
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    if sum(i.file_size for i in archive.infolist()) > 100 * 1024 * 1024:
                        raise ValueError("Expanded document exceeds the 100 MB limit.")
            target = unique_upload_path(upload_dir, name)
            target.write_bytes(content)
            if not chunk_documents(load_documents(target, index.data_dir)):
                raise ValueError("No readable text found. Scanned PDFs require OCR before upload.")
            results.append({"name": target.name, "ok": True})
        except Exception as exc:
            if target is not None:
                target.unlink(missing_ok=True)
            detail = str(exc) if isinstance(exc, ValueError) else "Could not read this document. Check that it is valid and unencrypted."
            results.append({"name": name, "ok": False, "error": detail})
    index.rebuild(progress=progress)
    return {"uploaded": results, "status": index.stats()}


@app.delete("/api/generic/documents/{name}")
def delete_document(name: str, request: Request) -> dict:
    index, _ = workspace_resources(request)
    if Path(name).name != name or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid document name.")
    target = index.data_dir / "uploads" / name
    if not target.is_file():
        raise HTTPException(404, "Document not found.")
    target.unlink()
    index.rebuild()
    # Deleted evidence must not remain in the answer cache.
    (request.state.workspace / "responses.json").unlink(missing_ok=True)
    (request.state.workspace / "questions.json").unlink(missing_ok=True)
    return index.stats()


def unique_upload_path(directory: Path, filename: str) -> Path:
    base = Path(filename).stem or "document"
    suffix = Path(filename).suffix
    candidate = directory / filename
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base}-{counter}{suffix}"
        counter += 1
    return candidate






@app.post("/api/generic/chat")
def generic_chat(request: ChatRequest, http_request: Request) -> dict:
    generic_rag, generic_response_cache = workspace_resources(http_request)
    if not generic_rag.ready:
        raise HTTPException(409, "Upload a readable document before asking a question.")
    interpretation = interpret_question(request.message, ResponseCache(http_request.state.workspace / 'intents.json'),
                                        history=[turn.model_dump() for turn in request.history],
                                        documents=generic_rag.stats()['sources'])
    response_language = interpretation['response_language']
    retrieval_query = interpretation['search_query']
    if interpretation['needs_clarification']:
        return {'message':request.message, 'answer':interpretation['clarification'] or 'Could you rephrase your question?',
                'citations':[], 'mode':'clarification', 'model':'', 'status':generic_rag.stats(),
                'interpretation':interpretation, 'retrieval_query':retrieval_query, 'response_language':response_language}
    cache_key = cache_key_parts("generic-chat", request, retrieval_query, response_language, generic_rag)
    cached = generic_response_cache.get(cache_key)
    if cached:
        return cached
    hits = generic_rag.search(retrieval_query, limit=request.top_k, phrase=interpretation["phrase"], all_meanings=True, search_phrases=interpretation.get("search_phrases", []))
    response = synthesize_answer(retrieval_query, hits, response_language=response_language)
    payload = {
        "message": request.message,
        "answer": response["answer"],
        "citations": response["citations"],
        "mode": response.get("mode"),
        "model": response.get("model"),
        "warning": response.get("warning"),
        "retrieval_query": retrieval_query,
        "interpretation": interpretation,
        "meaning_coverage": getattr(generic_rag, "meaning_coverage", None),
        "response_language": response_language,
        "status": generic_rag.stats(),
    }
    return generic_response_cache.set(cache_key, payload)


@app.post("/api/generic/compare")
def generic_compare(request: CompareRequest, http_request: Request) -> dict:
    if request.providers or request.models:
        raise HTTPException(400, "Use the approved OpenAI model selection for this comparison.")
    selected = request.openai_models or ['gpt-5-nano', 'gpt-6-luna', 'gpt-6.1-sol']
    if len(set(selected)) != len(selected):
        raise HTTPException(400, "Choose different models for the comparison.")
    request = request.model_copy(update={'openai_models':selected})
    generic_rag, generic_response_cache = workspace_resources(http_request)
    if not generic_rag.ready:
        raise HTTPException(409, "Upload a readable document first.")
    interpretation = interpret_question(request.message, ResponseCache(http_request.state.workspace / 'intents.json'),
                                        history=[turn.model_dump() for turn in request.history],
                                        documents=generic_rag.stats()['sources'])
    response_language = interpretation['response_language']
    retrieval_query = interpretation['search_query']
    providers = request.providers or ["openai", "groq"]
    cache_request = request.model_copy(update={"providers": providers})
    cache_key = cache_key_parts("generic-compare", cache_request, retrieval_query, response_language, generic_rag)
    cached = generic_response_cache.get(cache_key)
    if cached:
        return cached
    hits = generic_rag.search(retrieval_query, limit=request.top_k, phrase=interpretation["phrase"], all_meanings=True, search_phrases=interpretation.get("search_phrases", []))
    try:
        response = compare_answers(retrieval_query, hits, request.openai_models, response_language)
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    payload = {
        "message": request.message,
        "results": response["results"],
        "comparison_budget": response.get("comparison_budget"),
        "citations": response["citations"],
        "retrieval_query": retrieval_query,
        "interpretation": interpretation,
        "meaning_coverage": getattr(generic_rag, "meaning_coverage", None),
        "response_language": response_language,
        "status": generic_rag.stats(),
        "synthesis": synthesis_status(),
    }
    return generic_response_cache.set(cache_key, payload)






def cache_key_parts(
    route: str,
    request: CompareRequest | ChatRequest,
    retrieval_query: str,
    response_language: str,
    index: RagIndex,
) -> dict:
    selected_index = index
    return {
        "route": route,
        "message": " ".join(request.message.lower().split()),
        "retrieval_query": " ".join(retrieval_query.lower().split()),
        "response_language": response_language,
        "top_k": request.top_k,
        "retrieval_version": "faiss-component-coverage-v11",
        "history": [turn.model_dump() for turn in request.history],
        "providers": getattr(request, "providers", None),
        "models": getattr(request, "models", None),
        "openai_models": getattr(request, "openai_models", None),
        "fingerprint": selected_index.stats()["fingerprint"],
        "synthesis": synthesis_status()["models"],
    }






from .battery import router as battery_router
app.include_router(battery_router)
