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
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .security import GuardrailMiddleware
from .rag.config import env_sources_for_status
from .rag.cache import ResponseCache
from .rag.index import RagIndex
from .rag.questions import generate_questions, generate_document_questions
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
DATA_DIR = ROOT / "data_njsharing"
GENERIC_DATA_DIR = ROOT / "data_generic"
STATIC_DIR = ROOT / "static"
INDEX_PATH = ROOT / "storage" / "rag_index.pkl"
GENERIC_INDEX_PATH = ROOT / "storage" / "generic_rag_index.pkl"
CACHE_PATH = ROOT / "storage" / "response_cache.json"
GENERIC_CACHE_PATH = ROOT / "storage" / "generic_response_cache.json"

rag = RagIndex(DATA_DIR, INDEX_PATH)
generic_rag = RagIndex(GENERIC_DATA_DIR, GENERIC_INDEX_PATH)
response_cache = ResponseCache(CACHE_PATH)
generic_response_cache = ResponseCache(GENERIC_CACHE_PATH)
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


@app.on_event("startup")
def startup() -> None:
    rag.load_or_build()
    generic_rag.load_or_build()


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "generic.html")


@app.get("/njsharing")
def njsharing_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/generic")
def generic_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "generic.html")


@app.get("/generic/compare")
def generic_compare_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "generic_compare.html")


@app.get("/njsharing/compare")
def compare_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "compare.html")


@app.get("/njsharing/freemodels")
def free_models_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "freemodels.html")


@app.get("/api/status")
def status() -> dict:
    stats = rag.stats()
    stats["env_sources"] = env_sources_for_status(ENV_SOURCES)
    stats["synthesis"] = synthesis_status()
    return stats


@app.get("/api/questions")
def questions() -> dict:
    return {"questions": generate_questions(rag.chunks, limit=100)}


@app.post("/api/reindex")
def reindex() -> dict:
    rag.rebuild()
    return rag.stats()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    return await upload_to_index(files, rag, DATA_DIR)


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


async def upload_to_index(files: list[UploadFile], index: RagIndex, data_dir: Path) -> dict:
    uploaded: list[dict] = []
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        original_name = Path(file.filename or "document").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            uploaded.append({"name": original_name, "ok": False, "error": f"Unsupported file type: {suffix or 'none'}"})
            continue
        target = unique_upload_path(upload_dir, original_name)
        content = await file.read()
        target.write_bytes(content)
        uploaded.append({"name": original_name, "ok": True, "path": str(target.relative_to(data_dir)), "bytes": len(content)})
    index.rebuild()
    return {"uploaded": uploaded, "status": index.stats()}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    response_language = detect_language(request.message)
    retrieval_query = translate_query_for_retrieval(request.message)
    if not rag.is_in_scope(retrieval_query):
        return out_of_scope_response(request.message)
    cache_key = cache_key_parts("chat", request, retrieval_query, response_language)
    cached = response_cache.get(cache_key)
    if cached:
        return cached
    hits = rag.search(retrieval_query, limit=request.top_k)
    response = synthesize_answer(request.message, hits, response_language=response_language)
    payload = {
        "message": request.message,
        "answer": response["answer"],
        "citations": response["citations"],
        "mode": response.get("mode"),
        "model": response.get("model"),
        "warning": response.get("warning"),
        "retrieval_query": retrieval_query,
        "response_language": response_language,
        "status": rag.stats(),
    }
    return response_cache.set(cache_key, payload)


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


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict:
    response_language = detect_language(request.message)
    retrieval_query = translate_query_for_retrieval(request.message)
    if not rag.is_in_scope(retrieval_query):
        return out_of_scope_compare_response(request.message)
    cache_key = cache_key_parts("compare", request, retrieval_query, response_language)
    cached = response_cache.get(cache_key)
    if cached:
        return cached
    hits = rag.search(retrieval_query, limit=request.top_k)
    response = compare_models(request.message, hits, request.providers, response_language=response_language)
    payload = {
        "message": request.message,
        "results": response["results"],
        "citations": response["citations"],
        "retrieval_query": retrieval_query,
        "response_language": response_language,
        "status": rag.stats(),
        "synthesis": synthesis_status(),
    }
    return response_cache.set(cache_key, payload)


@app.post("/api/freemodels")
def free_models(request: CompareRequest) -> dict:
    response_language = detect_language(request.message)
    retrieval_query = translate_query_for_retrieval(request.message)
    if not rag.is_in_scope(retrieval_query):
        return out_of_scope_compare_response(request.message)
    cache_key = cache_key_parts("freemodels", request, retrieval_query, response_language)
    cached = response_cache.get(cache_key)
    if cached:
        return cached
    hits = rag.search(retrieval_query, limit=request.top_k)
    response = compare_free_models(request.message, hits, request.models, response_language=response_language)
    payload = {
        "message": request.message,
        "results": response["results"],
        "citations": response["citations"],
        "retrieval_query": retrieval_query,
        "response_language": response_language,
        "status": rag.stats(),
        "synthesis": synthesis_status(),
    }
    return response_cache.set(cache_key, payload)


def cache_key_parts(
    route: str,
    request: CompareRequest | ChatRequest,
    retrieval_query: str,
    response_language: str,
    index: RagIndex | None = None,
) -> dict:
    selected_index = index or rag
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


def out_of_scope_response(message: str) -> dict:
    return {
        "message": message,
        "answer": "This question appears to be outside the indexed NJ Sharing Network documents. Ask about the loaded HR SOPs, HIPAA notice, policies, procedures, benefits, badges, privacy, or related document content.",
        "citations": [],
        "mode": "scope-gate",
        "model": "none",
        "warning": "Skipped retrieval and model calls because the query did not match the indexed document scope.",
        "status": rag.stats(),
        "cache": {"hit": False, "tokens_saved": 0},
    }


def out_of_scope_compare_response(message: str) -> dict:
    return {
        "message": message,
        "results": [
            {
                "mode": "scope-gate",
                "model": "none",
                "ok": False,
                "answer": "",
                "error": "Skipped retrieval and model calls because the query appears outside the indexed document scope.",
                "elapsed_seconds": 0,
                "tokens": {"input": 0, "output": 0, "estimated": False},
                "pricing": {"input_per_million": 0, "output_per_million": 0},
                "estimated_cost_usd": 0,
            }
        ],
        "citations": [],
        "status": rag.stats(),
        "synthesis": synthesis_status(),
        "cache": {"hit": False, "tokens_saved": 0},
    }


from .battery import router as battery_router
app.include_router(battery_router)


@app.get('/compare', include_in_schema=False)
def legacy_compare_redirect():
    return RedirectResponse('/njsharing/compare', status_code=307)


@app.get('/freemodels', include_in_schema=False)
def legacy_free_models_redirect():
    return RedirectResponse('/njsharing/freemodels', status_code=307)
