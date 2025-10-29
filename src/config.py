from pathlib import Path

# Paths
DATA_DIR = Path("data")
VECTORSTORE_DIR = Path("vectorstore")

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL = "google/flan-t5-small"

# Chunking
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 3
SIM_THRESHOLD = 0.25  # filter out weak matches

# Toggle whether to load/use the cross-encoder reranker. Set to False for
# low-memory or Cloud environments that may fail when loading the reranker.
USE_RERANKER = False
USE_RERANKER = True
