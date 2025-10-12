import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, GENERATION_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SIM_THRESHOLD
)

# ----------------- File Readers -----------------

def _read_text_file(fp: Path) -> str:
    return fp.read_text(encoding="utf-8")

def _read_csv_faqs(fp: Path) -> List[str]:
    df = pd.read_csv(fp)
    rows = []
    if "question" in df.columns and "answer" in df.columns:
        for _, r in df.iterrows():
            rows.append(f"Q: {r['question']}\nA: {r['answer']}")
    return rows

def _read_xlsx_faqs(fp: Path) -> List[str]:
    df = pd.read_excel(fp)
    rows = []
    if "question" in df.columns and "answer" in df.columns:
        for _, r in df.iterrows():
            rows.append(f"Q: {r['question']}\nA: {r['answer']}")
    return rows

# ----------------- Corpus Loader -----------------


def load_corpus(data_dir: Path) -> List[Dict[str, Any]]:
    corpus = []
    for p in data_dir.glob("**/*"):
        if p.is_dir():
            continue

        # Case 1: plain text or phrasebook
        if p.suffix.lower() == ".txt":
            lines = p.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                if "|" in line:  # phrasebook style with separators
                    parts = [seg.strip() for seg in line.split("|")]
                    if len(parts) == 3:
                        text = f"English: {parts[0]}\nFe'efe'e: {parts[1]}\nFrench: {parts[2]}"
                    else:
                        text = line.strip()
                else:
                    text = line.strip()
                corpus.append({
                    "id": f"{p}#{i}",
                    "text": text,
                    "meta": {"source": f"{p.name} (line {i})"}
                })

        # Case 2: CSV FAQ with Q/A columns
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            if "question" in df.columns and "answer" in df.columns:
                for idx, r in enumerate(df.itertuples(index=False), start=1):
                    q = str(getattr(r, "question", "")).strip()
                    a = str(getattr(r, "answer", "")).strip()
                    if q and a:
                        text = f"Q: {q}\nA: {a}"
                        corpus.append({
                            "id": f"{p}#{idx}",
                            "text": text,
                            "meta": {"source": f"{p.name} (row {idx})"}
                        })

        # Case 3: Excel FAQ with Q/A columns
        elif p.suffix.lower() == ".xlsx":
            df = pd.read_excel(p)
            if "question" in df.columns and "answer" in df.columns:
                for idx, r in enumerate(df.itertuples(index=False), start=1):
                    q = str(getattr(r, "question", "")).strip()
                    a = str(getattr(r, "answer", "")).strip()
                    if q and a:
                        text = f"Q: {q}\nA: {a}"
                        corpus.append({
                            "id": f"{p}#{idx}",
                            "text": text,
                            "meta": {"source": f"{p.name} (row {idx})"}
                        })

        else:
            print(f"⚠️ Skipped unsupported file: {p}")

    return corpus

# ----------------- Chunking -----------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
        if i <= 0:
            break
    return chunks

def build_documents(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for item in corpus:
        chunks = chunk_text(item["text"])
        for idx, ch in enumerate(chunks):
            docs.append({
                "id": f"{item['id']}::chunk{idx}",
                "text": ch,
                "meta": item["meta"]
            })
    return docs

# ----------------- Vector Index -----------------

class VectorIndex:
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.docs: List[Dict[str, Any]] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def build(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        vectors = self._embed([d["text"] for d in docs]).astype("float32")
        d = vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[int, float]]:
        q = self._embed([query]).astype("float32")
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def save(self, vs_dir: Path):
        vs_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(vs_dir / "faiss.index"))
        with open(vs_dir / "docs.pkl", "wb") as f:
            pickle.dump(self.docs, f)

    def load(self, vs_dir: Path):
        self.index = faiss.read_index(str(vs_dir / "faiss.index"))
        with open(vs_dir / "docs.pkl", "rb") as f:
            self.docs = pickle.load(f)

# ----------------- Generator -----------------

class Generator:
    def __init__(self, model_name: str = GENERATION_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------- Prompt -----------------

def format_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return f"""You are a banking assistant.
Question: {query}

No sources were found. Answer: I could not find this in the documents."""

    cited = "\n\n".join(
        [f"[{i+1}] Source: {c['meta']['source']}\n{c['text']}" for i, c in enumerate(contexts)]
    )
    return f"""You are a banking assistant. 
Use ONLY the sources provided below to answer the question.
If the answer is unclear or missing, say "I could not find this in the documents."

Question: {query}

Sources:
{cited}

Answer (with citations like [1], [2]):"""

# ----------------- RAG Pipeline -----------------

# class RAGPipeline:
#     def __init__(self, vs_dir: Path = VECTORSTORE_DIR):
#         self.vs_dir = vs_dir
#         self.index = VectorIndex()
#         self.generator = Generator()

#     def ensure_loaded(self):
#         if not (self.vs_dir / "faiss.index").exists():
#             raise FileNotFoundError("Vector store not found. Run ingest.py first.")
#         self.index.load(self.vs_dir)

#     def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
#         results = self.index.search(query, k)
#         # Apply similarity filter
#         filtered = []
#         for i, s in results:
#             if s >= SIM_THRESHOLD:
#                 filtered.append(self.index.docs[i] | {"score": s})
#         return filtered

#     def answer(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
#         if not contexts:
#             return {
#                 "answer": "I could not find this in the documents.",
#                 "contexts": [],
#                 "prompt": f"Question: {query}\n\nNo strong matches found."
#             }

#         # ✅ Case 1: FAQ-style Q&A
#         for i, c in enumerate(contexts):
#             text = c["text"]
#             if text.strip().startswith("Q:") and "A:" in text:
#                 qna_parts = text.split("A:", 1)
#                 if len(qna_parts) == 2:
#                     ans = qna_parts[1].strip()
#                     return {
#                         "answer": f"{ans} [{i+1}]",
#                         "contexts": contexts,
#                         "prompt": f"Direct FAQ match.\n\nQuestion: {query}"
#                     }

#         # ✅ Case 2: Phrasebook style (multi-language → extract only Fe'efe'e)
#         for i, c in enumerate(contexts):
#             text = c["text"]
#             if "Fe'efe'e:" in text:
#                 feefee = text.split("Fe'efe'e:")[1]
#                 if "French:" in feefee:
#                     feefee = feefee.split("French:")[0]
#                 answer = feefee.strip()
#                 return {
#                     "answer": f"{answer} [{i+1}]",
#                     "contexts": contexts,
#                     "prompt": f"Direct phrasebook match.\n\nQuestion: {query}"
#                 }

#         # ✅ Case 3: Fallback → use generator
#         prompt = format_prompt(query, contexts)
#         completion = self.generator.generate(prompt)

#         if not completion.strip() or "I could not find" in completion:
#             completion = "I could not find this in the documents."

#         return {
#             "answer": completion,
#             "contexts": contexts,
#             "prompt": prompt
#         }

import re

class RAGPipeline:
    def __init__(self, vs_dir: Path = VECTORSTORE_DIR):
        self.vs_dir = vs_dir
        self.index = VectorIndex()
        self.generator = Generator()

    def ensure_loaded(self):
        if not (self.vs_dir / "faiss.index").exists():
            raise FileNotFoundError("Vector store not found. Run ingest.py first.")
        self.index.load(self.vs_dir)

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        results = self.index.search(query, k)
        filtered = []
        for i, s in results:
            if s >= SIM_THRESHOLD:
                filtered.append(self.index.docs[i] | {"score": s})
        return filtered

    def answer(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not contexts:
            return {
                "answer": "I could not find this in the documents.",
                "contexts": [],
                "prompt": f"Question: {query}\n\nNo strong matches found."
            }

        # ✅ Case 1: FAQ-style Q&A
        for i, c in enumerate(contexts):
            text = c["text"]
            if text.strip().startswith("Q:") and "A:" in text:
                qna_parts = text.split("A:", 1)
                if len(qna_parts) == 2:
                    ans = qna_parts[1].strip()
                    return {
                        "answer": f"{ans} [{i+1}]",
                        "contexts": contexts,
                        "prompt": f"Direct FAQ match.\n\nQuestion: {query}"
                    }

        # ✅ Case 2: Phrasebook style (translation → extract only Fe’efe’e)
        if ("how do you say" in query.lower() 
            or "translate" in query.lower() 
            or re.search(r"[éèêàùç]", query)):  # detect French input
            for i, c in enumerate(contexts):
                text = c["text"]
                if "Fe'efe'e:" in text:
                    feefee = text.split("Fe'efe'e:")[1]
                    if "French:" in feefee:
                        feefee = feefee.split("French:")[0]
                    return {
                        "answer": f"{feefee.strip()} [{i+1}]",
                        "contexts": contexts,
                        "prompt": f"Direct phrasebook match.\n\nQuestion: {query}"
                    }

        # ✅ Case 3: Keyword fallback for French queries
        for i, c in enumerate(contexts):
            if "French:" in c["text"]:
                french_text = c["text"].split("French:")[1].lower()
                if any(word in french_text for word in query.lower().split()):
                    if "Fe'efe'e:" in c["text"]:
                        feefee = c["text"].split("Fe'efe'e:")[1]
                        if "French:" in feefee:
                            feefee = feefee.split("French:")[0]
                        return {
                            "answer": f"{feefee.strip()} [{i+1}]",
                            "contexts": contexts,
                            "prompt": f"Keyword fallback match.\n\nQuestion: {query}"
                        }

        # ✅ Case 4: Generator fallback
        prompt = format_prompt(query, contexts)
        completion = self.generator.generate(prompt)

        if not completion.strip() or "I could not find" in completion:
            completion = "I could not find this in the documents."

        return {
            "answer": completion,
            "contexts": contexts,
            "prompt": prompt
        }
