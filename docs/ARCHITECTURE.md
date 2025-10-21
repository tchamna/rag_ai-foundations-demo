ARCHITECTURE
============

Overview
--------
The project is organized into a small set of focused components:

- `src/app_streamlit.py` — Streamlit UI. Handles uploads, rebuild requests, user queries, transcript export and shows a runtime health banner.
- `src/rag_pipeline.py` — Retrieval and generation logic. Contains `VectorIndex` (FAISS wrapper), optional `embedder` and `generator` lazy loaders, and search/re-rank glue.
- `src/ingest.py` and `src/precompute_embeddings.py` — Document ingestion and local index creation. Use these scripts to generate the `vectorstore/` artifacts that can be deployed.
- `vectorstore/` — Output of the ingestion step: a FAISS index file and metadata pickles. Deploy this directory with your code for lightweight runtime.

Design choices
--------------
- Precomputed vectorstore by default: avoids heavy ML dependencies in production. The code includes a lexical-overlap fallback when embeddings are unavailable so the app stays functional (lower quality) even if embeddings cannot be computed on-demand.
- Lazy model loading: when embeddings or generators are requested the pipeline attempts to import and initialize them. This reduces cold-start overhead in simple deployments.
- Defensive UI: the Streamlit layer catches exceptions from the pipeline, logs them to a runtime temp file (`rag_runtime_errors.log`) and shows a friendly error message so the UI doesn't disappear on uncaught exceptions.

Health & telemetry
------------------
- The UI displays a runtime banner showing whether the `vectorstore/` loaded and whether `sentence-transformers` is available. That is intended to make cloud deployments easier to debug at a glance.
- For reproducible debugging, the app writes error tracebacks to the system temp directory and also logs to stderr. On Azure App Service the runtime temp paths are usually under `D:\local\Temp` or `D:\home\LogFiles\Application`.

Extending the system
--------------------
- Replace the local generator with an API-backed LLM (OpenAI, Azure OpenAI, Anthropic) by implementing a generator adapter in `src/rag_pipeline.py`.
- Add authentication and rate-limiting in front of the FastAPI or Streamlit UI for production usage.

