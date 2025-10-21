from pathlib import Path
import sys
import argparse

# Ensure repository root is on sys.path so imports like `from src.rag_pipeline` work
# whether the script is launched from the repo root or from inside `src/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.rag_pipeline import load_corpus, build_documents, VectorIndex
    from src.config import DATA_DIR, VECTORSTORE_DIR
except ImportError as e:
    print("Missing dependency when importing project modules:", e)
    print("Please install dependencies, e.g.:")
    print("  pip install -r requirements.txt")
    print("Then re-run this script: python src\\precompute_embeddings.py")
    sys.exit(1)


def run(data_dir: Path = DATA_DIR, vs_dir: Path = VECTORSTORE_DIR):
    print(f"Loading corpus from {data_dir}")
    corpus = load_corpus(data_dir)
    print(f"Loaded {len(corpus)} files")
    docs = build_documents(corpus)
    print(f"Split into {len(docs)} chunks")

    idx = VectorIndex()
    print("Building vector index (this will load embeddings locally)")
    idx.build(docs)
    print("Saving vectorstore")
    idx.save(vs_dir)
    print(f"Saved vectorstore to {vs_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute embeddings for RAG pipeline")
    parser.add_argument('--data_dir', type=Path, default=DATA_DIR, help='Directory containing the data files')
    parser.add_argument('--vs_dir', type=Path, default=VECTORSTORE_DIR, help='Directory to save the vectorstore')
    args = parser.parse_args()
    run(data_dir=args.data_dir, vs_dir=args.vs_dir)
