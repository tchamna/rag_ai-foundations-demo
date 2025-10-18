# RAG AI Foundations Demo

This repository is a small Retrieval-Augmented Generation (RAG) demo that combines FAISS vector search, SentenceTransformers embeddings, and text generation (local FLAN-T5 or ChatGPT). It includes:

- A Streamlit dashboard (`src/app_streamlit.py`) for interactive Q&A and transcript downloads (XLSX).
- A FastAPI backend and React frontend (in `frontend/`) for a separate web chat UI.
- Scripts for ingesting documents and precomputing embeddings.
- A FAISS vectorstore saved under `vectorstore/`.

This README covers quick local usage and notes about the UI and theme.

## Quickstart — Streamlit (local)

1. Create a virtual environment and install requirements (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit dashboard:

```powershell
streamlit run src/app_streamlit.py
```

3. In the Streamlit UI:
- Use the **Upload Document** sidebar to add `.txt`, `.csv` or `.xlsx` files.
- Click **Rebuild Vector Store** in the sidebar (or use the `Build vector store now` button in the right column) to index documents.
- Ask questions in the main panel. Retrieved chunks and the Vector DB Browser appear in the right column.
- Download the Q&A transcript using the **Download Transcript (XLSX)** button (XLSX only — CSV removed for better Excel compatibility).

Notes:
- The app uses a centralized theme provider in `src/theme.py`. Dark-mode is enabled by default; you can change the default in `src/app_streamlit.py` by modifying the `st.session_state['dark_mode']` initialization.

## Quickstart — Docker (backend + frontend)

The repo includes Dockerfiles and a `docker-compose.yml` to run the backend (FastAPI) and the React frontend together.

From the project root, build and run with Docker Compose:

```powershell
docker-compose up --build
```

After startup:
- Backend (FastAPI / Uvicorn) typically listens on port `8000` inside the compose network.
- Frontend (Vite preview) is served at port `5173`.

If you run the frontend locally (not via Docker) make sure API calls target the backend service URL (e.g. `http://localhost:8000` when running backend locally).

## Rebuild Vector Store (programmatic)

The vector store build logic lives in `src/app_streamlit.py` (function `rebuild_index()`), and uses the ingestion helper functions from `rag_pipeline.py`. Building will create files under the configured `VECTORSTORE_DIR` (see `src/config.py`).

If you want to precompute embeddings separately, check `src/precompute_embeddings.py` and `src/ingest.py`.

## Theme / Styling

- Theme CSS is centralized in `src/theme.py` as `get_theme_css(dark_mode: bool)`. The Streamlit app imports this and applies it at startup.
- Dark mode is the default. Toggle it in the sidebar to switch back to light mode.

## Notes & Caveats

- The Streamlit app may lazy-load models (FLAN-T5/transformers) when used; expect extra memory and time on the first query.
- Large dependencies such as `torch` and ML models can take time to install and initialize.
- The transcript download is XLSX only to avoid cross-platform encoding issues with CSV/Excel.

## Development

- Code is organized under `src/` for the Python backends and `frontend/` for the React UI.
- To run tests or linters, add your preferred tooling and CI configs.

---
If you'd like, I can:
- Commit and push this README to the repository for you, or
- Add a short `Makefile` / `tasks.json` to streamline common commands (run, build, rebuild vector store).
# Banking Assistant — RAG Demo (Portfolio)

This project is a compact, demonstrable Retrieval-Augmented Generation (RAG) system I built to showcase practical skills with NLP, embeddings, vector search, and LLM-driven answer synthesis. It's designed to be friendly: clear architecture, polished UX (Streamlit), and easy to run locally or deploy to the Cloud.

Why this project is portfolio-worthy
- Real-world scenario: customer-facing Q&A over banking FAQs, product disclosures, and uploaded bank statements.
- Full RAG stack: document ingestion, chunking, SentenceTransformer embeddings, FAISS vector search, and LLM-based answer generation with grounding citations.
- Reproducible: runnable locally and tuned for Streamlit Cloud deployment.
- Thoughtful UX: transcript download, retrieval browser, configurable similarity threshold, and optional ChatGPT polishing.

![App interface](assets/image.png)

Live demo (local or cloud)
- Run locally with CPU-only models or deploy to Streamlit Cloud for quick sharing with anyone, as a prototype.
- Hosted demo (Streamlit Cloud): https://tchamna-rag-ai-foundation-model.streamlit.app/

Key features
- Ingest text/csv/xlsx and automatically chunk documents for retrieval.
- FAISS vector store backed by SentenceTransformers embeddings (`all-MiniLM-L6-v2` by default).
- Local generation with `google/flan-t5-base` (no external API keys required) and optional ChatGPT polishing.
- Phrasebook-aware shortcuts: direct answers for Q/A and bilingual phrasebook entries (Fe'efe'e / English / French).
- Streamlit UI with controls for similarity threshold, reranker, and transcript download.

Custom documents & any domain
- You can customize the app to search over your own documents anytime. Drop new .txt/.csv/.xlsx files in the `data/docs/` folder or use the "Upload Document" area in the app sidebar.
- After adding or removing documents, rebuild (recompute embeddings / retrain) the vector store by running:
```powershell
python src/ingest.py --data_dir data/docs --vs_dir vectorstore
```
or click the **Rebuild Vector Store** button in the sidebar of the running app.
- Once the vector store is rebuilt the app will search over your new content. This means the system works for any domain (legal, product manuals, HR, phrasebooks, etc.) — as long as you update the corpus and rebuild the index, the retrieval and generation will use your context.


Quickstart
1) Create & activate a Python environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies (Streamlit Cloud-ready)
```powershell
pip install -r requirements.txt
```

3) Build the vector store (one-time step or after uploads)
```powershell
python src/ingest.py --data_dir data/docs --vs_dir vectorstore
```

4) Run the app
```powershell
streamlit run src/app_streamlit.py

```
if that doesnt work, do

python -m streamlit run src/app_streamlit.py

Open the URL printed by Streamlit (typically http://localhost:8501) and try example prompts.

Example prompts to try
- "What are overdraft fees and how can I avoid them?"
- "Summarize recurring charges in the uploaded statement for May."
- "How do I say 'I love you' in Fe'efe'e?"

Repository layout
```
.  # root
├─ README.md
├─ requirements.txt
├─ data/docs/                # source documents for ingestion
├─ vectorstore/              # index (created by ingest.py)
└─ src/
   ├─ app_streamlit.py       # Streamlit interface
   ├─ ingest.py              # corpus builder + index creation
   ├─ rag_pipeline.py        # retrieval + answer generation
   └─ config.py
```

Notes for reviewers
- This demo is intentionally self-contained and modest in compute requirements. Replace the local generator with a production LLM (OpenAI, Bedrock, etc.) and add auth/guardrails for production-readiness.
- For deployment to Streamlit Cloud, see the pinned `requirements.txt` (below).


Contact / attribution
- Repo owner: Shck Tchamna

---
