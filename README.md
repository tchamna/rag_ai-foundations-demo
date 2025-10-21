# RAG AI Foundations Demo

This repository is a small Retrieval-Augmented Generation (RAG) demo that combines FAISS vector search, SentenceTransformers embeddings, and text generation (local FLAN-T5 or ChatGPT). It includes:

- A Streamlit dashboard (`src/app_streamlit.py`) for interactive Q&A and transcript downloads (XLSX).
- A FastAPI backend and React frontend (in `frontend/`) for a separate web chat UI.
# RAG AI Foundations Demo

This repository is a compact Retrieval-Augmented Generation (RAG) demo built around FAISS vector search and a Streamlit UI. The app prefers using a precomputed FAISS index (in `vectorstore/`) so it can run in lightweight runtimes without heavy ML dependencies. When embeddings or generators are required at runtime the code will attempt to lazy-load them — but production deployments should precompute and commit the vector store instead of installing large ML packages.

Contents
- Streamlit dashboard: `src/app_streamlit.py` (interactive Q&A, transcript export)
- RAG pipeline & ingestion: `src/rag_pipeline.py`, `src/ingest.py`, `src/precompute_embeddings.py`
- Frontend: `frontend/` (separate React UI)
- Precomputed vector store: `vectorstore/` (FAISS index + metadata)

This README contains a short quickstart. Full usage, deployment and architecture notes are in the `docs/` folder.

Quick local setup
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install runtime requirements (this project keeps ML-heavy packages out of `requirements.txt` by default; add them only if you need on-demand embedding/generation):

```powershell
pip install -r requirements.txt
```

3. (Optional) Precompute the vectorstore locally and commit it to the repo (recommended for cloud deploys):

```powershell
python src/precompute_embeddings.py --data_dir data/docs --vs_dir vectorstore
```

4. Run the Streamlit app (the repo uses `app.py` as the Streamlit entrypoint):

```powershell
streamlit run app.py --server.port 8501
```

If your shell requires the module form:

```powershell
python -m streamlit run app.py
```

What to do in the app
- Upload documents via the sidebar (supported: .txt, .csv, .xlsx).
- If you uploaded new documents, rebuild the vector store from the sidebar.
- Ask questions in the main panel. The app will show retrieved passages and citations.
- Download the transcript using the Download Transcript button; the app falls back to CSV if openpyxl isn't available in the runtime.

Important runtime notes
- The app is designed to run with a precomputed `vectorstore/` so cloud deployments do not need heavy ML packages (torch, sentence-transformers). If you remove the precomputed vectorstore and the runtime cannot install ML packages, querying will fall back to a lexical overlap search but quality will be lower.
- A runtime health banner is shown at the top of the Streamlit UI. It reports:
   - whether `vectorstore/` loaded successfully, and
   - whether the `sentence_transformers` package is present (used only when on-demand embeddings are required).
- For XLSX transcript export we use `openpyxl` when available; otherwise the app falls back to CSV and logs the exception to a runtime temp file for diagnosis.

Where to find more documentation
- `docs/USAGE.md` — extended usage and examples
- `docs/DEPLOY_AZURE.md` — instructions and the exact startup command for Azure App Service
- `docs/ARCHITECTURE.md` — high-level design and component notes

Contributing
- If you add new data or rebuild the vector store locally, include the `vectorstore/` artifacts in your deployment bundle so the cloud app can run without ML dependencies.

Troubleshooting tips (quick)
- If you see `ModuleNotFoundError: No module named 'sentence_transformers'` it means the runtime is missing the embedding package — either precompute the vectorstore or add `sentence-transformers` + `torch` to your deployment requirements (not recommended for constrained App Service plans).
- If you see `ModuleNotFoundError: No module named 'xml'` while importing `openpyxl`, check for a shadowing PyPI package named `xml` in `site-packages` (run `pip list` and look for a top-level package named `xml`) and remove it. The docs in `docs/DEPLOY_AZURE.md` include Kudu commands to inspect the runtime.

License & attribution
- Repo owner: Shck Tchamna

---

See the `docs/` folder for more details. If you'd like, I can also add a `Makefile` or VS Code tasks to standardize the run/build workflow.

## Scoring and Ranking

This application returns multiple numeric signals for every retrieved context. Understanding these will help you interpret results and tune the app's behavior.

- `score` (semantic score)
   - Source: FAISS index (or lexical-overlap fallback when embeddings are not available).
   - Meaning: a measure of semantic similarity (or token overlap) between query and chunk. When embeddings are normalized the app maps FAISS values into a user-friendly range for display. Higher is better.
   - When used: the default relevance metric when no reranker is enabled.

- `rerank_score` (cross-encoder / reranker)
   - Source: an optional CrossEncoder-style reranker (SentenceTransformers CrossEncoder) that scores (query, context) pairs.
   - Meaning: a model-dependent scalar (often raw logits) which can be negative or positive. Larger values indicate higher predicted relevance. Many reranker models can output negative logits, but a positive value is typically a stronger relevance signal than a negative one — regardless, ranking compares these values relative to each other and selects the larger value as more relevant.
   - When used: if a reranker is enabled, the pipeline can use `rerank_score` instead of (or alongside) `score` to refine ordering.

- `final_score` (decision metric)
   - Source: computed by the retrieval pipeline according to the selected ranking strategy.
   - Meaning: a single numeric value used to order contexts and to choose the primary context for short-answer extraction.

Ranking strategies available in the app

- `semantic` — order by `score` only. Use this when you prefer vector-similarity to determine winners.
- `rerank` — prefer `rerank_score` when available; fall back to `score` when a context lacks a reranker value.
- `weighted` — normalize reranker outputs across returned contexts and combine them with semantic `score` using a tunable `rerank_weight` (0..1). This lets you keep semantic similarity as the dominant signal but let the reranker influence ties.
- `auto` — prefer reranker when it produced scores for the returned contexts; otherwise use semantic.

Practical tips

- If you deploy without a reranker (lighter runtime), use `semantic` or `auto`. The app will fall back to lexical matching when embeddings are unavailable.
- If you enable the reranker but observe surprising orderings, try `weighted` with a low `rerank_weight` (e.g., 0.2) so semantic score remains primary.
- Use the debug toggle in the Retrieved Chunks panel to show `score`, `rerank_score`, and `final_score` for each chunk so you can diagnose why a particular chunk was chosen.

Why a chunk with higher `score` might not be chosen

- If the reranker is enabled and assigns a much higher (less negative or larger) `rerank_score` to a chunk with a slightly lower semantic `score`, the reranker will make that chunk the `final_score` winner. In that case choose `semantic` or `weighted` with a low rerank weight if you want vector-similarity to win.

