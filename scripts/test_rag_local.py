import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_pipeline import RAGPipeline
from src.config import VECTORSTORE_DIR, TOP_K, USE_RERANKER


def quick_test(q: str = "How do I report a lost card?"):
    print("Starting quick RAG test...")
    rag = RAGPipeline(VECTORSTORE_DIR, use_chatgpt=False, use_reranker=USE_RERANKER)
    try:
        rag.ensure_loaded()
    except FileNotFoundError as e:
        print("Vectorstore not found:", e)
        return

    retrieved = rag.retrieve(q, TOP_K)
    ctxs = retrieved or []
    if not ctxs:
        docs = rag.index.docs
        idxs = [i for i, d in enumerate(docs) if q.lower() in d["text"].lower()]
        ctxs = [docs[i] for i in idxs[:TOP_K]]

    result = rag.answer(q, ctxs)
    print('\nQuestion:', q)
    print('\nAnswer:', result.get('answer'))
    print('\nMode:', result.get('mode'))
    print('\nTop contexts:')
    for i, c in enumerate(ctxs, 1):
        txt = c['text'].replace('\n', ' ')[:300]
        print(f" [{i}] {c['meta']['source']}: {txt}...")


if __name__ == '__main__':
    quick_test()
