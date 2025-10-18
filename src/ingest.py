import argparse
from pathlib import Path

from rag_pipeline import load_corpus, build_documents, VectorIndex
from src.config import DATA_DIR, VECTORSTORE_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Folder with .txt/.csv files")
    parser.add_argument("--vs_dir", type=str, default=str(VECTORSTORE_DIR), help="Folder to store FAISS index")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    vs_dir = Path(args.vs_dir)

    print(f"📂 Loading documents from: {data_dir}")
    corpus = load_corpus(data_dir)
    print(f"✅ Loaded {len(corpus)} documents")

    docs = build_documents(corpus)
    print(f"✂️ Split into {len(docs)} chunks")

    index = VectorIndex()
    index.build(docs)
    index.save(vs_dir)

    print(f"💾 Saved vectorstore with {len(docs)} chunks → {vs_dir}")
