"""Local sentence embeddings shared by all workspace FAISS indexes."""
import os
import threading
import numpy as np

MODEL_NAME = os.getenv('RAG_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
_lock = threading.RLock()
_model = None


def encode(texts):
    global _model
    from sentence_transformers import SentenceTransformer
    with _lock:
        if _model is None:
            import torch
            torch.set_num_threads(max(1, int(os.getenv("RAG_EMBEDDING_THREADS", "4"))))
            _model = SentenceTransformer(MODEL_NAME, device='cpu', trust_remote_code=False)
        return np.ascontiguousarray(_model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False), dtype='float32')
