# Deploying the Gradio RAG demo to Hugging Face Spaces

Steps to deploy:

1. Create a new Gradio Space on Hugging Face. Choose "Gradio" as the SDK.

2. Push this repository (or a copy) to the new Space's Git remote.

3. In the Space settings, set the "Hardware" to CPU/GPU depending on your needs.

4. Make sure `requirements-spaces.txt` is present in the repo root. Spaces will install it.

5. IMPORTANT: precompute embeddings locally and upload the `vectorstore/` folder to the Space repo
   (committing `vectorstore/faiss.index` and `vectorstore/docs.pkl`) so the Space does not need to build
   embeddings at startup. To precompute locally run:

    ```bash
    python src\precompute_embeddings.py
    git add vectorstore/faiss.index vectorstore/docs.pkl
    git commit -m "Add precomputed vectorstore"
    git push
    ```

6. Push the repo to the Space. The entrypoint `app.py` will start the configured demo (update the Space entrypoint as needed).

Notes:
- If `faiss` cannot be built on Spaces, commit only `docs.pkl` and the app will fall back to lexical search.
- Keep `requirements-spaces.txt` minimal to reduce installation time on Spaces.
