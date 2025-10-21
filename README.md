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
