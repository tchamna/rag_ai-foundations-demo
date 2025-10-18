from pathlib import Path
import sys

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
    run()
