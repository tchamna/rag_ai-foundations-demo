# Web chat UI (React + FastAPI)

This repository now includes a lightweight web chat UI built with React (Vite) that talks to a FastAPI backend exposing the RAG pipeline.

Run the backend (Python):

```powershell
# from project root
python -m pip install -r requirements.txt
uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000
```

Run the frontend (in a separate terminal):

```powershell
cd frontend
npm install
npm run dev
```

The frontend will open at http://localhost:5173 and send queries to http://localhost:8000.

Notes:
- The Streamlit dashboard is unchanged and remains available as `streamlit run src/app_streamlit.py`.
- The React frontend uses the `/query`, `/rebuild`, and `/transcript` endpoints on the FastAPI server.
