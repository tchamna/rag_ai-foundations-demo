USAGE
=====

This file contains extended usage examples and practical tips.

1) Local quickstart (recommended)

- Create and activate a Python venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

- Install dependencies

```powershell
pip install -r requirements.txt
```

- Precompute vectorstore (recommended for cloud deploys):

```powershell
python src/precompute_embeddings.py --data_dir data/docs --vs_dir vectorstore
```

- Run Streamlit

```powershell
streamlit run app.py
```

2) Rebuilding the vectorstore from the UI

- Use the sidebar control "Rebuild Vector Store" after uploading documents.
- Or run the ingest script:

```powershell
python src/ingest.py --data_dir data/docs --vs_dir vectorstore
```

3) Export & transcripts

- Use the Download Transcript button. If `openpyxl` is not available the app will save a CSV fallback.
- If you rely on XLSX exports in production, add `openpyxl` to your `requirements.txt`.

4) Example prompts

- "What are overdraft fees and how can I avoid them?"
- "Summarize recurring charges in the uploaded statement for May."
- "How do I say 'I love you' in Fe'efe'e?"

5) Notes on embedding & generation

- The app supports two modes:
  - Precomputed vectorstore mode (recommended for cloud): no heavy ML packages required at runtime.
  - On-demand embedding/generation mode: requires `sentence-transformers`, `torch`, and optionally `transformers` for local generator models. Install these only when needed.

6) Troubleshooting

- "ModuleNotFoundError: No module named 'sentence_transformers'": Precompute the vectorstore or add the package to your deployment.
- "ModuleNotFoundError: No module named 'xml'" while importing `openpyxl`: check for a shadowing package named `xml` in `pip list` and remove it. See `docs/DEPLOY_AZURE.md` for Kudu commands.

