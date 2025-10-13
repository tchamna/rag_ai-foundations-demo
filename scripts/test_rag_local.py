from rag_pipeline import RAGPipeline
from config import VECTORSTORE_DIR, TOP_K, USE_RERANKER


def quick_test(q: str = "How to say 'I love you' in Fe'efe'e?"):
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
